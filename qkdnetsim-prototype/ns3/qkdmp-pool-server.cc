/*
 * qkdmp-pool-server: QKDNetSim as the key-pool engine of the qkdmp Level-B DES.
 *
 * Builds one QKD site per topology node -- a QKD module node, a KMS node and a
 * QKDN controller -- and, per topology edge, a QKD link: a post-processing
 * application pair that generates key at the edge's rate and stores it (over
 * HTTP) in the Q-buffers of both end KMSs. The Python DES keeps all control
 * logic (requests, attacker, detector, policies, outcomes); this program only
 * answers pool queries, so the same policy code runs against QKDNetSim's key
 * supply instead of the fluid pool model.
 *
 * Protocol (one command per stdin line, one reply per stdout line). Times are
 * DES seconds; QKDNetSim time is preroll + t.
 *   SYNC t          advance to t, reply "L b_0 ... b_{E-1}" (usable bits per edge)
 *   TAKE t e bits   advance to t, remove ceil(bits / keySize) keys from both ends
 *                   of edge e (only if both ends hold them), reply "T taken_bits"
 *   RAW t           advance to t without trimming, reply "R u_0:v_0 ... u_{E-1}:v_{E-1}"
 *                   (bits in the Q-buffer at each end)
 *                   (for measuring the realised key rate with no consumption)
 *   ATTACK pDrop pTamper k n_1 .. n_k
 *                   set the compromised relays and their drop / tamper probabilities
 *   SEND t sid k n_0 .. n_{k-1} hex
 *                   at time t, send share `sid` (payload bytes as hex) from n_0 to
 *                   n_{k-1} along the given path, hop by hop (see "Share relay")
 *   RECV t sid_1 .. sid_m
 *                   advance to t, reply "D sid,status,seen,hex ..." per share:
 *                   status A (arrived) / X (dropped) / P (still in flight), seen 1
 *                   if a compromised relay handled the share in plaintext
 *   INFO            reply "I missing_keys sent_packets"
 *   QUIT
 *
 * Share relay: every KMS runs a UDP relay endpoint on its KMS-KMS links. A share
 * crosses each hop one-time-pad encrypted with key material that the DES
 * reserved (TAKE) on that link: TAKE keeps the removed keys, and each hop uses
 * one of them (both ends of a QKD link hold the same key, so the receiver
 * decrypts with it). Each relay decrypts, processes for `hopProcessing` seconds,
 * re-encrypts with the next link's key and forwards. A compromised relay sees
 * the plaintext; it drops the share with probability pDrop, otherwise replaces
 * it with random bytes with probability pTamper (qkdmp.attacker.share_outcome).
 *
 * Topology file: first line "E keySizeBits", then one line per edge
 * "u v rate_bps capacity_bits length_km". Node ids are 0..N-1.
 *
 * Pool capacity: QKDNetSim configures one Mmax for every Q-buffer, so the
 * per-edge capacity is enforced by trimming both Q-buffers to capacity at every
 * query. Between two queries nothing is consumed, so trimming then equals
 * refusing keys that arrive while the buffer is full.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/q-buffer.h"
#include "ns3/qkd-app-helper.h"
#include "ns3/qkd-key-manager-system-application.h"
#include "ns3/qkd-link-helper.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

struct Edge
{
    uint32_t u, v;
    double rateBps, capacityBits, lengthKm;
    Ptr<QBuffer> bufU, bufV; // Q-buffer at KMS(u) towards KMS(v), and vice versa
};

std::vector<Edge> g_edges;
uint32_t g_keySize = 256;
double g_preroll = 2.0;
std::vector<Ptr<Node>> g_km;

// ---- share relay state ----
const uint16_t RELAY_PORT = 9000;
double g_hopProcessing = 1e-4;
std::map<std::pair<uint32_t, uint32_t>, size_t> g_edgeIndex;   // (u, v) either order -> edge
std::vector<std::pair<Ipv4Address, Ipv4Address>> g_kmLinkAddr; // edge -> (addr at u, addr at v)
std::map<size_t, std::vector<std::string>> g_reserved;          // edge -> reserved key strings
std::vector<Ptr<Socket>> g_relaySocket;                         // per KMS node
std::set<uint32_t> g_compromised;
double g_pDrop = 0.0, g_pTamper = 0.0;
Ptr<UniformRandomVariable> g_rv;
uint64_t g_missingKeys = 0, g_sentPackets = 0;

struct Share
{
    std::vector<uint32_t> path;
    std::vector<std::string> hopKey; // key used on hop i (path[i] -> path[i+1])
    std::string payload;             // plaintext at the destination
    char status = 'P';
    bool seen = false;
};

std::map<uint64_t, Share> g_shares;

Ptr<QKDKeyManagerSystemApplication>
GetKms(Ptr<Node> node)
{
    for (uint32_t i = 0; i < node->GetNApplications(); ++i)
    {
        Ptr<QKDKeyManagerSystemApplication> kms =
            DynamicCast<QKDKeyManagerSystemApplication>(node->GetApplication(i));
        if (kms)
        {
            return kms;
        }
    }
    NS_FATAL_ERROR("no KMS on node " << node->GetId());
    return nullptr;
}

void
ResolveBuffers()
{
    for (auto& e : g_edges)
    {
        if (!e.bufU)
        {
            e.bufU = GetKms(g_km[e.u])->GetQBuffer(g_km[e.v]->GetId());
        }
        if (!e.bufV)
        {
            e.bufV = GetKms(g_km[e.v])->GetQBuffer(g_km[e.u]->GetId());
        }
    }
}

void
AdvanceTo(double t)
{
    Time target = Seconds(g_preroll + t);
    if (target > Simulator::Now())
    {
        Simulator::Stop(target - Simulator::Now());
        Simulator::Run();
    }
    ResolveBuffers();
}

void
Trim(Ptr<QBuffer> buf, double capacityBits)
{
    while (buf && buf->GetBitCount() > capacityBits && buf->GetKeyCount() > 0)
    {
        buf->GetKey("");
    }
}

uint32_t
Usable(const Edge& e)
{
    if (!e.bufU || !e.bufV)
    {
        return 0;
    }
    return std::min(e.bufU->GetBitCount(), e.bufV->GetBitCount());
}

std::string
Xor(const std::string& data, const std::string& key)
{
    std::string out(data);
    for (size_t i = 0; i < out.size(); ++i)
    {
        out[i] = static_cast<char>(out[i] ^ (key.empty() ? 0 : key[i % key.size()]));
    }
    return out;
}

std::string
ToHex(const std::string& s)
{
    static const char* d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s)
    {
        out += d[c >> 4];
        out += d[c & 15];
    }
    return out;
}

std::string
FromHex(const std::string& h)
{
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2)
    {
        out += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
    }
    return out;
}

std::string
PopReservedKey(size_t edge)
{
    auto& keys = g_reserved[edge];
    if (keys.empty())
    {
        ++g_missingKeys; // should not happen: every hop was reserved by the DES
        return std::string();
    }
    std::string k = keys.back();
    keys.pop_back();
    return k;
}

void SendHop(uint64_t sid, size_t hop, std::string plaintext);

// Share `sid` is at path[hop] in plaintext (after decryption at this relay).
void
AtNode(uint64_t sid, size_t hop, std::string plaintext)
{
    Share& sh = g_shares[sid];
    uint32_t node = sh.path[hop];
    if (hop + 1 == sh.path.size())
    {
        sh.payload = plaintext;
        sh.status = 'A';
        return;
    }
    if (hop > 0 && g_compromised.count(node))
    {
        sh.seen = true;
        if (g_rv->GetValue() < g_pDrop)
        {
            sh.status = 'X';
            return;
        }
        if (g_rv->GetValue() < g_pTamper)
        {
            for (auto& c : plaintext)
            {
                c = static_cast<char>(g_rv->GetInteger(0, 255));
            }
        }
    }
    Simulator::Schedule(Seconds(hop > 0 ? g_hopProcessing : 0.0), &SendHop, sid, hop, plaintext);
}

void
SendHop(uint64_t sid, size_t hop, std::string plaintext)
{
    Share& sh = g_shares[sid];
    uint32_t a = sh.path[hop], b = sh.path[hop + 1];
    size_t e = g_edgeIndex.at({a, b});
    std::string key = PopReservedKey(e);
    sh.hopKey[hop] = key;
    std::string msg = std::to_string(sid) + " " + std::to_string(hop + 1) + " " + ToHex(Xor(plaintext, key));
    Ipv4Address to = (g_edges[e].u == b) ? g_kmLinkAddr[e].first : g_kmLinkAddr[e].second;
    Ptr<Packet> pkt = Create<Packet>(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
    g_relaySocket[a]->SendTo(pkt, 0, InetSocketAddress(to, RELAY_PORT));
    ++g_sentPackets;
}

void
OnRelayReceive(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    while ((pkt = socket->Recv()))
    {
        std::string msg(pkt->GetSize(), '\0');
        pkt->CopyData(reinterpret_cast<uint8_t*>(&msg[0]), msg.size());
        std::istringstream in(msg);
        uint64_t sid;
        size_t hop;
        std::string hex;
        in >> sid >> hop >> hex;
        auto it = g_shares.find(sid);
        if (it == g_shares.end())
        {
            continue;
        }
        AtNode(sid, hop, Xor(FromHex(hex), it->second.hopKey[hop - 1]));
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string topoFile;
    uint32_t seed = 1;
    double linkDelayMs = 0.1;
    uint64_t ppDataRate = 1000; // as in the QKDNetSim SECOQC example
    uint32_t blockBits = 0;     // post-processing key block; 0: one pool key
    CommandLine cmd(__FILE__);
    cmd.AddValue("topo", "topology file", topoFile);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("preroll", "QKDNetSim seconds before DES time 0 (pools fill up)", g_preroll);
    cmd.AddValue("ppDataRate",
                 "post-processing classical traffic rate (bps)",
                 ppDataRate);
    cmd.AddValue("blockBits",
                 "bits per post-processing key block; the KMS splits it into pool keys (0: one key)",
                 blockBits);
    cmd.AddValue("hopProcessing", "relay processing time per hop (s)", g_hopProcessing);
    cmd.AddValue("localDelayMs", "delay of the QKD-module <-> KMS link (ms)", linkDelayMs);
    cmd.Parse(argc, argv);

    // The KMS moves key from a link's Q-buffer into two local S-buffers
    // (encryption / decryption) up to SMaximal. With no S-buffer room all
    // key stays in the Q-buffer, which is then the link's whole pool.
    Config::SetDefault("ns3::SBuffer::SMinimal", UintegerValue(0));
    Config::SetDefault("ns3::SBuffer::SThreshold", UintegerValue(0));
    Config::SetDefault("ns3::SBuffer::SMaximal", UintegerValue(0));

    RngSeedManager::SetSeed(1 + seed);
    RngSeedManager::SetRun(1);
    srand(seed + 1);

    std::ifstream in(topoFile);
    if (!in)
    {
        NS_FATAL_ERROR("cannot open topology file " << topoFile);
    }
    size_t nEdges;
    in >> nEdges >> g_keySize;
    uint32_t nNodes = 0;
    for (size_t i = 0; i < nEdges; ++i)
    {
        Edge e;
        in >> e.u >> e.v >> e.rateBps >> e.capacityBits >> e.lengthKm;
        nNodes = std::max(nNodes, std::max(e.u, e.v) + 1);
        g_edges.push_back(e);
    }

    // Nodes: QKD modules 0..N-1, KMSs N..2N-1, controllers 2N..3N-1.
    NodeContainer qkd, km, ctrl;
    qkd.Create(nNodes);
    km.Create(nNodes);
    ctrl.Create(nNodes);
    InternetStackHelper internet;
    internet.Install(qkd);
    internet.Install(km);
    internet.Install(ctrl);
    for (uint32_t v = 0; v < nNodes; ++v)
    {
        g_km.push_back(km.Get(v));
    }

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.252");

    // Module <-> own KMS (store_key traffic), and KMS <-> KMS per edge.
    std::vector<Ipv4Address> kmAddr(nNodes);
    for (uint32_t v = 0; v < nNodes; ++v)
    {
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(linkDelayMs)));
        Ipv4InterfaceContainer ic = ipv4.Assign(p2p.Install(qkd.Get(v), km.Get(v)));
        ipv4.NewNetwork();
        kmAddr[v] = ic.GetAddress(1);
    }
    // QKD link classical channel (sifting / post-processing), delay = length / 2e5 km/s.
    std::vector<std::pair<Ipv4Address, Ipv4Address>> linkAddr;
    for (auto& e : g_edges)
    {
        p2p.SetChannelAttribute("Delay", TimeValue(Seconds(e.lengthKm / 2.0e5)));
        Ipv4InterfaceContainer ic = ipv4.Assign(p2p.Install(qkd.Get(e.u), qkd.Get(e.v)));
        ipv4.NewNetwork();
        linkAddr.emplace_back(ic.GetAddress(0), ic.GetAddress(1));
        Ipv4InterfaceContainer kk = ipv4.Assign(p2p.Install(km.Get(e.u), km.Get(e.v)));
        ipv4.NewNetwork();
        g_kmLinkAddr.emplace_back(kk.GetAddress(0), kk.GetAddress(1));
        g_edgeIndex[{e.u, e.v}] = g_kmLinkAddr.size() - 1;
        g_edgeIndex[{e.v, e.u}] = g_kmLinkAddr.size() - 1;
    }

    QKDAppHelper appHelper;
    QKDLinkHelper linkHelper;
    std::vector<Ptr<QKDControl>> controllers;
    for (uint32_t v = 0; v < nNodes; ++v)
    {
        controllers.push_back(linkHelper.InstallQKDNController(ctrl.Get(v)));
    }
    // Q-buffers: no floor, unbounded here (capacity is per edge, see Trim);
    // S-buffers get no room so the KMS never moves key out of the Q-buffers.
    linkHelper.ConfigureQBuffers(controllers, 0, 0, 500000000, g_keySize);
    linkHelper.ConfigureRSBuffers(controllers, 0, 0, 0, g_keySize);
    for (uint32_t v = 0; v < nNodes; ++v)
    {
        appHelper.InstallKeyManager(km.Get(v), kmAddr[v], 80, controllers[v]);
    }

    ApplicationContainer pp;
    for (size_t i = 0; i < g_edges.size(); ++i)
    {
        const Edge& e = g_edges[i];
        // Post-processing sockets bind to Any:port, so every link at a node
        // needs its own port.
        uint16_t port = static_cast<uint16_t>(1000 + i);
        pp.Add(appHelper.InstallPostProcessing(qkd.Get(e.u),
                                               qkd.Get(e.v),
                                               InetSocketAddress(linkAddr[i].first, port),
                                               InetSocketAddress(linkAddr[i].second, port),
                                               ctrl.Get(e.u),
                                               ctrl.Get(e.v),
                                               (blockBits ? blockBits : g_keySize) / 8,
                                               DataRate(static_cast<uint64_t>(e.rateBps)),
                                               100,
                                               DataRate(ppDataRate)));
    }
    pp.Start(Seconds(0));
    pp.Stop(Seconds(1e6));

    // share relay endpoints, one per KMS
    g_rv = CreateObject<UniformRandomVariable>();
    for (uint32_t v = 0; v < nNodes; ++v)
    {
        Ptr<Socket> sock = Socket::CreateSocket(km.Get(v), UdpSocketFactory::GetTypeId());
        sock->Bind(InetSocketAddress(Ipv4Address::GetAny(), RELAY_PORT));
        sock->SetRecvCallback(MakeCallback(&OnRelayReceive));
        g_relaySocket.push_back(sock);
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    linkHelper.CreateTopologyGraph(controllers);
    linkHelper.PopulateRoutingTables();

    std::string line;
    std::cout << "READY " << g_edges.size() << std::endl;
    while (std::getline(std::cin, line))
    {
        std::istringstream ss(line);
        std::string op;
        ss >> op;
        if (op == "SYNC")
        {
            double t;
            ss >> t;
            AdvanceTo(t);
            std::ostringstream out;
            out << "L";
            for (auto& e : g_edges)
            {
                Trim(e.bufU, e.capacityBits);
                Trim(e.bufV, e.capacityBits);
                out << ' ' << Usable(e);
            }
            std::cout << out.str() << std::endl;
        }
        else if (op == "TAKE")
        {
            double t, bits;
            size_t idx;
            ss >> t >> idx >> bits;
            AdvanceTo(t);
            Edge& e = g_edges.at(idx);
            Trim(e.bufU, e.capacityBits);
            Trim(e.bufV, e.capacityBits);
            uint32_t keys = static_cast<uint32_t>((bits + g_keySize - 1) / g_keySize);
            uint32_t taken = 0;
            if (Usable(e) >= keys * g_keySize)
            {
                for (uint32_t k = 0; k < keys; ++k)
                {
                    Ptr<QKDKey> a = e.bufU->GetKey("");
                    Ptr<QKDKey> b = e.bufV->GetKey("");
                    if (a && b)
                    {
                        taken += g_keySize;
                        g_reserved[idx].push_back(a->GetKeyString()); // kept for the share relay
                    }
                }
            }
            std::cout << "T " << taken << std::endl;
        }
        else if (op == "RAW")
        {
            double t;
            ss >> t;
            AdvanceTo(t);
            std::ostringstream out;
            out << "R";
            for (auto& e : g_edges)
            {
                out << ' ' << (e.bufU ? e.bufU->GetBitCount() : 0) << ':'
                    << (e.bufV ? e.bufV->GetBitCount() : 0);
            }
            std::cout << out.str() << std::endl;
        }
        else if (op == "ATTACK")
        {
            size_t k;
            ss >> g_pDrop >> g_pTamper >> k;
            g_compromised.clear();
            for (size_t i = 0; i < k; ++i)
            {
                uint32_t n;
                ss >> n;
                g_compromised.insert(n);
            }
            std::cout << "OK" << std::endl;
        }
        else if (op == "SEND")
        {
            double t;
            uint64_t sid;
            size_t k;
            ss >> t >> sid >> k;
            Share sh;
            sh.path.resize(k);
            for (size_t i = 0; i < k; ++i)
            {
                ss >> sh.path[i];
            }
            std::string hex;
            ss >> hex;
            sh.hopKey.resize(k > 0 ? k - 1 : 0);
            AdvanceTo(t);
            g_shares[sid] = sh;
            Simulator::ScheduleNow(&AtNode, sid, static_cast<size_t>(0), FromHex(hex));
            std::cout << "OK" << std::endl;
        }
        else if (op == "RECV")
        {
            double t;
            ss >> t;
            AdvanceTo(t);
            std::ostringstream out;
            out << "D";
            uint64_t sid;
            while (ss >> sid)
            {
                auto it = g_shares.find(sid);
                if (it == g_shares.end())
                {
                    out << ' ' << sid << ",U,0,-";
                    continue;
                }
                const Share& sh = it->second;
                out << ' ' << sid << ',' << sh.status << ',' << (sh.seen ? 1 : 0) << ','
                    << (sh.status == 'A' ? ToHex(sh.payload) : "-");
                if (sh.status != 'P')
                {
                    g_shares.erase(it);
                }
            }
            std::cout << out.str() << std::endl;
        }
        else if (op == "INFO")
        {
            std::cout << "I " << g_missingKeys << ' ' << g_sentPackets << std::endl;
        }
        else if (op == "QUIT")
        {
            break;
        }
        else
        {
            std::cout << "E unknown command" << std::endl;
        }
    }
    Simulator::Destroy();
    return 0;
}
