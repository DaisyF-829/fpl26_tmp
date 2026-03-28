#include "heterographs.h"
#include "globals.h"
#include "timing_info.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <climits>

#include "tatum/analyzers/SetupTimingAnalyzer.hpp"
#include "tatum/analyzers/HoldTimingAnalyzer.hpp"
#include "tatum/analyzers/SetupHoldTimingAnalyzer.hpp"
#include "tatum/report/TimingReportTagRetriever.hpp"
#include "tatum/timing_paths.hpp"
#include "tatum/util/tatum_strong_id.hpp"
#include "tatum/TimingGraph.hpp"
#include "tatum/TimingConstraints.hpp"
#include "tatum/report/timing_path_tracing.hpp"
#include "tatum/error.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include "cnpy.h"
#include "atom_delay_calc.h"

namespace tgnn
{

// =============================================================================
// Internal helper: get the BLIF net name for a tnode (for debug annotations)
// =============================================================================
static std::string get_blif_or_default(const AtomContext&  atom_ctx,
                                        tatum::NodeId       tn,
                                        const std::string&  no_net_placeholder,
                                        const std::string&  no_pin_placeholder,
                                        bool*               has_real_net_out = nullptr)
{
    AtomPinId apin = atom_ctx.lookup.tnode_atom_pin(tn);
    if (!apin) {
        if (has_real_net_out) *has_real_net_out = false;
        return no_pin_placeholder;
    }
    AtomNetId net = atom_ctx.nlist.pin_net(apin);
    if (!net) {
        if (has_real_net_out) *has_real_net_out = false;
        return no_net_placeholder;
    }
    if (has_real_net_out) *has_real_net_out = true;
    return atom_ctx.nlist.net_name(net);
}

// =============================================================================
// Step 1 – build tnode / tedge skeleton from timing graph (called pre-placement)
// =============================================================================
void HeteroGraph::build_from_timing_graph_prepl()
{
    const auto& timing_ctx = g_vpr_ctx.timing();
    const auto& tg         = timing_ctx.graph;

    // Build tnode list
    for (const tatum::NodeId node_id : tg->nodes()) {
        Tnode* tnode = new Tnode{node_id, tg->node_type(node_id)};
        tnode_nodes.emplace_back(tnode);
        tnode_id_map[node_id]  = tnode_nodes.size() - 1;
        tnode_ptr_map[node_id] = tnode;
    }

    // Build tedge list (delay values for non-INTERCONNECT filled here;
    // INTERCONNECT delays are filled later by enrich_edge_features)
    const auto& atom_ctx = g_vpr_ctx.atom();
    AtomDelayCalc atom_delay_calc(atom_ctx.nlist, atom_ctx.lookup);

    for (const tatum::EdgeId edge_id : tg->edges()) {
        tatum::NodeId   src_id    = tg->edge_src_node(edge_id);
        tatum::NodeId   dst_id    = tg->edge_sink_node(edge_id);
        tatum::EdgeType edge_type = tg->edge_type(edge_id);

        // Annotate BLIF names (debug only)
        auto upsert = [&](tatum::NodeId tn) {
            if (tnode_blif_name.count(tn)) return;
            bool real = false;
            tnode_blif_name.emplace(tn, get_blif_or_default(atom_ctx, tn,
                                        no_net_placeholder, no_pin_placeholder, &real));
            tnode_has_real_net.emplace(tn, real);
        };
        upsert(src_id);
        upsert(dst_id);

        if (edge_type == tatum::EdgeType::PRIMITIVE_CLOCK_CAPTURE) {
            AtomPinId clk_pin = atom_ctx.lookup.tnode_atom_pin(src_id);
            AtomPinId dat_pin = atom_ctx.lookup.tnode_atom_pin(dst_id);
            float t = (clk_pin && dat_pin)
                          ? atom_delay_calc.atom_setup_time(clk_pin, dat_pin)
                          : 0.0f;
            tnode_edges.emplace_back(new TEdge{src_id, dst_id, edge_type, t});
            continue;
        }
        if (edge_type == tatum::EdgeType::PRIMITIVE_CLOCK_LAUNCH) {
            AtomPinId clk_pin = atom_ctx.lookup.tnode_atom_pin(src_id);
            AtomPinId out_pin = atom_ctx.lookup.tnode_atom_pin(dst_id);
            float d = atom_delay_calc.atom_clock_to_q_delay(clk_pin, out_pin, DelayType::MAX);
            tnode_edges.emplace_back(new TEdge{src_id, dst_id, edge_type, d});
            continue;
        }
        if (edge_type == tatum::EdgeType::PRIMITIVE_COMBINATIONAL) {
            AtomPinId src_pin = atom_ctx.lookup.tnode_atom_pin(src_id);
            AtomPinId snk_pin = atom_ctx.lookup.tnode_atom_pin(dst_id);
            float d = atom_delay_calc.atom_combinational_delay(src_pin, snk_pin, DelayType::MAX);
            tnode_edges.emplace_back(new TEdge{src_id, dst_id, edge_type, d});
            continue;
        }

        // INTERCONNECT edge – defer delay; record net/ipin for later lookup
        // We still need to find the RR endpoints so we reuse extract_used_rrnodes
        // logic inline here via a lightweight path.
        TEdge* e = new TEdge{src_id, dst_id, edge_type, 0.0f};
        e->has_net_delay = false;
        tnode_edges.emplace_back(e);
    }
}

// =============================================================================
// Internal helper – fills has_net_delay / parent_net_id / ipin / src_rrnode /
// dst_rrnode for INTERCONNECT edges, and populates used_rrnodes.
// =============================================================================
void HeteroGraph::extract_used_rrnodes(const Netlist<>& netlist)
{
    const auto& timing_ctx      = g_vpr_ctx.timing();
    const auto& tg              = timing_ctx.graph;
    const auto& rr_graph        = g_vpr_ctx.device().rr_graph;
    const auto& atom_ctx        = g_vpr_ctx.atom();
    const auto& net_rr_terminals = g_vpr_ctx.routing().net_rr_terminals;

    // Build a map from (src_tnode, dst_tnode) -> TEdge* for INTERCONNECT
    std::unordered_map<tatum::NodeId,
        std::unordered_map<tatum::NodeId, TEdge*>> interconnect_edges;
    for (TEdge* e : tnode_edges) {
        if (e->edge_type == tatum::EdgeType::INTERCONNECT)
            interconnect_edges[e->src_tnode][e->dst_tnode] = e;
    }

    for (const tatum::EdgeId edge_id : tg->edges()) {
        if (tg->edge_type(edge_id) != tatum::EdgeType::INTERCONNECT) continue;

        tatum::NodeId src_id = tg->edge_src_node(edge_id);
        tatum::NodeId dst_id = tg->edge_sink_node(edge_id);

        auto it1 = interconnect_edges.find(src_id);
        if (it1 == interconnect_edges.end()) continue;
        auto it2 = it1->second.find(dst_id);
        if (it2 == it1->second.end()) continue;
        TEdge* edge = it2->second;

        // --- locate cluster net and ipin ---
        AtomPinId src_atom_pin = atom_ctx.lookup.tnode_atom_pin(src_id);
        if (!src_atom_pin) continue;
        AtomBlockId src_atom_blk = atom_ctx.nlist.pin_block(src_atom_pin);
        ClusterBlockId src_clb   = atom_ctx.lookup.atom_clb(src_atom_blk);
        if (!src_clb.is_valid()) continue;

        AtomPinId sink_atom_pin = atom_ctx.lookup.tnode_atom_pin(dst_id);
        if (!sink_atom_pin) continue;
        AtomBlockId sink_atom_blk = atom_ctx.nlist.pin_block(sink_atom_pin);
        ClusterBlockId clb_sink   = atom_ctx.lookup.atom_clb(sink_atom_blk);

        const t_pb_graph_pin* sink_gpin = atom_ctx.lookup.atom_pin_pb_graph_pin(sink_atom_pin);
        if (!sink_gpin) continue;
        size_t sink_pb_route_id = sink_gpin->pin_count_in_cluster;

        size_t         sink_net_pin_index  = (size_t)-1;
        ClusterNetId   sink_cluster_net_id = ClusterNetId::INVALID();
        size_t         sink_block_pin_index;
        std::tie(sink_cluster_net_id, sink_block_pin_index, sink_net_pin_index) =
            find_pb_route_clb_input_net_pin(clb_sink, sink_pb_route_id);

        if (sink_cluster_net_id == ClusterNetId::INVALID() ||
            sink_net_pin_index == (size_t)-1) continue;

        if (netlist.net_is_ignored(sink_cluster_net_id)) continue;

        RRNodeId src_rr_id  = net_rr_terminals[sink_cluster_net_id][0];
        RRNodeId sink_rr_id = net_rr_terminals[sink_cluster_net_id][sink_net_pin_index];

        // Cache RRNode metadata
        auto make_rr_node = [&](RRNodeId rr_id) -> RRNode* {
            auto it = used_rrnodes.find(rr_id);
            if (it != used_rrnodes.end()) {
                it->second->usage_count++;
                return it->second;
            }
            RRNode* n = new RRNode{
                rr_id,
                size_t(rr_graph.node_type(rr_id)),
                (size_t)rr_graph.node_xlow(rr_id),
                (size_t)rr_graph.node_xhigh(rr_id),
                (size_t)rr_graph.node_ylow(rr_id),
                (size_t)rr_graph.node_yhigh(rr_id),
                rr_graph.node_R(rr_id),
                rr_graph.node_C(rr_id),
                1};
            used_rrnodes[rr_id] = n;
            return n;
        };

        edge->src_rrnode    = make_rr_node(src_rr_id);
        edge->dst_rrnode    = make_rr_node(sink_rr_id);
        edge->has_net_delay = true;
        edge->parent_net_id = sink_cluster_net_id;
        edge->ipin          = sink_net_pin_index;
    }
}

// =============================================================================
// Post-route: refresh RR endpoints / tile context for INTERCONNECT edges
// =============================================================================
void HeteroGraph::initialize_tile_map()
{
    const auto& device_ctx = g_vpr_ctx.device();
    grid_w                 = device_ctx.grid.width();
    grid_h                 = device_ctx.grid.height();
}

void HeteroGraph::prune_rr_graph()
{
    // Placeholder: full BFS prune can be reintroduced when RR–tile graph is wired.
}

void HeteroGraph::update_pruned_tnode_rr_edges()
{
    // Placeholder: sync TEdge RR pointers after prune.
}

void HeteroGraph::build_rr_node_map()
{
    initialize_tile_map();

    for (auto& kv : used_rrnodes) {
        delete kv.second;
    }
    used_rrnodes.clear();

    for (TEdge* e : tnode_edges) {
        if (e->edge_type != tatum::EdgeType::INTERCONNECT) continue;
        e->src_rrnode    = nullptr;
        e->dst_rrnode    = nullptr;
        e->has_net_delay = false;
    }

    const Netlist<>& netlist = (const Netlist<>&)g_vpr_ctx.clustering().clb_nlist;
    extract_used_rrnodes(netlist);
    prune_rr_graph();
    update_pruned_tnode_rr_edges();
}

// =============================================================================
// Placement STA → pl_arrival_time (same tag logic as post-route arrival)
// =============================================================================
void HeteroGraph::extract_arrival_time_preroute(const SetupTimingInfo& timing_info)
{
    const tatum::SetupTimingAnalyzer& analyzer = *timing_info.setup_analyzer();

    for (auto& tnode : tnode_nodes) {
        auto tags = analyzer.setup_tags(tnode->id, tatum::TagType::DATA_ARRIVAL);
        if (tags.empty()) continue;
        tnode->has_pl_arrival_time = true;
        float max_arr              = -std::numeric_limits<float>::infinity();
        for (const auto& tag : tags)
            max_arr = std::max(max_arr, (float)tag.time());
        tnode->pl_arrival_time = max_arr;
    }
}

// =============================================================================
// Step 2 – fill rt_arrival_time / rt_required_time / rt_slack labels
// =============================================================================
void HeteroGraph::extract_arrival_time_aftroute(const SetupTimingInfo& timing_info)
{
    const tatum::SetupTimingAnalyzer& analyzer = *timing_info.setup_analyzer();

    for (auto& tnode : tnode_nodes) {
        // DATA_ARRIVAL: max over all launch-domain tags
        {
            auto tags = analyzer.setup_tags(tnode->id, tatum::TagType::DATA_ARRIVAL);
            if (!tags.empty()) {
                tnode->has_rt_arrival_time = true;
                float max_arr = -std::numeric_limits<float>::infinity();
                for (const auto& tag : tags)
                    max_arr = std::max(max_arr, (float)tag.time());
                tnode->rt_arrival_time = max_arr;
            }
        }

        // DATA_REQUIRED: max (least-constrained) required time over all capture-domain tags
        {
            auto tags = analyzer.setup_tags(tnode->id, tatum::TagType::DATA_REQUIRED);
            if (!tags.empty()) {
                tnode->has_rt_required = true;
                float max_req = -std::numeric_limits<float>::infinity();
                for (const auto& tag : tags)
                    max_req = std::max(max_req, (float)tag.time());
                tnode->rt_required_time = max_req;
            }
        }

        // Slack = required - arrival (valid only when both are available)
        if (tnode->has_rt_arrival_time && tnode->has_rt_required)
            tnode->rt_slack = tnode->rt_required_time - tnode->rt_arrival_time;
    }

    critical_path_delay = timing_info.longest_critical_path().delay();
}

// =============================================================================
// Step 2b – extract top-K longest-delay setup paths and mark critical nodes
//
// Strategy: find_critical_paths() returns only 1 path per clock-domain pair,
// which is insufficient when K > #domain_pairs.  Instead we iterate every
// logical-output (sink) node, collect all (sink, launch_domain, capture_domain,
// arrival_time) candidates, sort by DATA_ARRIVAL time descending, and trace
// the top-K via trace_setup_path().
// =============================================================================
void HeteroGraph::extract_top_k_paths(const SetupTimingInfo& timing_info, int K)
{
    const auto& timing_ctx  = g_vpr_ctx.timing();
    const tatum::TimingGraph&         tg       = *timing_ctx.graph;
    const tatum::SetupTimingAnalyzer& analyzer = *timing_info.setup_analyzer();

    // ---- 1. Collect candidates: one entry per (sink, launch_domain, capture_domain) ----
    struct Candidate {
        tatum::NodeId   sink;
        tatum::DomainId launch;
        tatum::DomainId capture;
        float           arrival;   // DATA_ARRIVAL at sink (raw seconds)
    };
    std::vector<Candidate> candidates;
    candidates.reserve(1024);

    for (tatum::NodeId node : tg.logical_outputs()) {
        // SLACK tags carry both launch and capture domains
        for (const tatum::TimingTag& stag : analyzer.setup_slacks(node)) {
            if (!stag.time().valid()) continue;
            tatum::DomainId ld = stag.launch_clock_domain();
            tatum::DomainId cd = stag.capture_clock_domain();

            // Find the DATA_ARRIVAL value at this sink for this launch domain
            float best_arr = -std::numeric_limits<float>::infinity();
            bool  found    = false;
            for (const tatum::TimingTag& atag :
                     analyzer.setup_tags(node, tatum::TagType::DATA_ARRIVAL)) {
                if (atag.launch_clock_domain() == ld) {
                    if ((float)atag.time() > best_arr) {
                        best_arr = (float)atag.time();
                        found    = true;
                    }
                }
            }
            if (!found) continue;
            candidates.push_back({node, ld, cd, best_arr});
        }
    }

    if (candidates.empty()) {
        std::cout << "[TGNN] extract_top_k_paths: no candidates found.\n";
        return;
    }

    // ---- 2. Sort by arrival time descending (longest-delay paths first) ----
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.arrival > b.arrival;
              });

    // ---- 3. Trace top-K paths ----
    int num_paths = std::min(K, (int)candidates.size());
    top_path_node_indices.reserve(num_paths);
    top_path_delays.reserve(num_paths);
    tnode_on_critical_path.assign(tnode_nodes.size(), 0);

    for (int p = 0; p < num_paths; ++p) {
        const Candidate& cand = candidates[p];
        if (!cand.sink.is_valid()) continue;

        tatum::TimingPath path = tatum::trace_setup_path(
            tg, analyzer,
            cand.launch, cand.capture, cand.sink);

        // Collect node indices from the data-arrival sub-path
        std::vector<int> node_idx_list;
        for (const tatum::TimingPathElem& elem : path.data_arrival_path().elements()) {
            auto it = tnode_id_map.find(elem.node());
            if (it == tnode_id_map.end()) continue;
            size_t idx = it->second;
            node_idx_list.push_back((int)idx);
            tnode_on_critical_path[idx] = 1;
        }

        if (node_idx_list.empty()) continue;

        top_path_node_indices.push_back(std::move(node_idx_list));
        // Store arrival time in units of 0.01ns (×1e11 matches other timing arrays)
        top_path_delays.push_back(cand.arrival * 1.0e11f);
    }

    std::cout << "[TGNN] extract_top_k_paths: traced "
              << top_path_node_indices.size() << " paths (requested " << K
              << " from " << candidates.size() << " candidates).\n";
}

// =============================================================================
// Step 3 – fill placement position, net HPWL/fanout, fanin/fanout per tnode
// =============================================================================

// Helper: map a tnode to its cluster block's placement coordinates
bool HeteroGraph::get_block_loc(tatum::NodeId node_id, int& x, int& y) const
{
    const auto& atom_ctx      = g_vpr_ctx.atom();
    const auto& placement_ctx = g_vpr_ctx.placement();

    AtomPinId atom_pin = atom_ctx.lookup.tnode_atom_pin(node_id);
    if (!atom_pin) return false;

    AtomBlockId   atom_blk = atom_ctx.nlist.pin_block(atom_pin);
    ClusterBlockId clb     = atom_ctx.lookup.atom_clb(atom_blk);
    if (!clb.is_valid()) return false;

    const auto& loc = placement_ctx.block_locs()[clb].loc;
    x = loc.x;
    y = loc.y;
    return true;
}

// Helper: compute HPWL and fanout of a cluster net
bool HeteroGraph::get_net_hpwl_fanout(ParentNetId net_id, float& hpwl, int& fanout) const
{
    const auto& cluster_ctx   = g_vpr_ctx.clustering();
    const auto& placement_ctx = g_vpr_ctx.placement();

    ClusterNetId clb_net{size_t(net_id)};
    if (!clb_net.is_valid()) return false;

    int x_min = INT_MAX, x_max = INT_MIN;
    int y_min = INT_MAX, y_max = INT_MIN;
    int pin_count = 0;

    for (ClusterPinId pin : cluster_ctx.clb_nlist.net_pins(clb_net)) {
        ClusterBlockId blk = cluster_ctx.clb_nlist.pin_block(pin);
        if (!blk.is_valid()) continue;
        const auto& loc = placement_ctx.block_locs()[blk].loc;
        x_min = std::min(x_min, loc.x);
        x_max = std::max(x_max, loc.x);
        y_min = std::min(y_min, loc.y);
        y_max = std::max(y_max, loc.y);
        ++pin_count;
    }

    if (pin_count < 2) return false;
    hpwl   = (float)((x_max - x_min) + (y_max - y_min));
    fanout = pin_count - 1;   // subtract the driver pin
    return true;
}

void HeteroGraph::extract_node_features()
{
    const auto& atom_ctx = g_vpr_ctx.atom();

    for (Tnode* t : tnode_nodes) {
        t->fanin = t->fanout = 0;
    }

    // ---- 1. Placement position + associated net features ----
    for (Tnode* tnode : tnode_nodes) {
        int x = -1, y = -1;
        if (!get_block_loc(tnode->id, x, y)) continue;
        tnode->blk_x = x;
        tnode->blk_y = y;

        // Retrieve the atom pin's net to compute net_hpwl / net_fanout
        AtomPinId   atom_pin = atom_ctx.lookup.tnode_atom_pin(tnode->id);
        if (!atom_pin) continue;
        AtomNetId   atom_net = atom_ctx.nlist.pin_net(atom_pin);
        if (!atom_net) continue;

        // Map atom net -> cluster net(s); use first valid one
        auto clb_nets_opt = atom_ctx.lookup.clb_nets(atom_net);
        if (!clb_nets_opt || clb_nets_opt->empty()) continue;

        for (ClusterNetId clb_net : *clb_nets_opt) {
            float hpwl   = -1.0f;
            int   fo     = -1;
            // Re-use the ParentNetId-based helper
            if (get_net_hpwl_fanout(ParentNetId(size_t(clb_net)), hpwl, fo)) {
                tnode->net_hpwl   = hpwl;
                tnode->net_fanout = fo;
                break;  // take first valid net
            }
        }
    }

    // ---- 2. Fanin / fanout from timing-graph topology ----
    for (const TEdge* e : tnode_edges) {
        auto it_src = tnode_ptr_map.find(e->src_tnode);
        auto it_dst = tnode_ptr_map.find(e->dst_tnode);
        if (it_src != tnode_ptr_map.end()) it_src->second->fanout++;
        if (it_dst != tnode_ptr_map.end()) it_dst->second->fanin++;
    }

    // ---- 3. Topological level (longest path from fanin-0 sources) ----
    for (Tnode* t : tnode_nodes)
        t->topo_level = -1;
    for (Tnode* t : tnode_nodes) {
        if (t->fanin == 0)
            t->topo_level = 0;
    }
    const int max_iter = std::max((int)tnode_nodes.size(), 1);
    for (int it = 0; it < max_iter; ++it) {
        bool changed = false;
        for (const TEdge* e : tnode_edges) {
            auto it_s = tnode_ptr_map.find(e->src_tnode);
            auto it_d = tnode_ptr_map.find(e->dst_tnode);
            if (it_s == tnode_ptr_map.end() || it_d == tnode_ptr_map.end()) continue;
            Tnode* s = it_s->second;
            Tnode* d = it_d->second;
            if (s->topo_level < 0) continue;
            int nl = s->topo_level + 1;
            if (d->topo_level < nl) {
                d->topo_level = nl;
                changed       = true;
            }
        }
        if (!changed) break;
    }

    std::cout << "[TGNN] extract_node_features: done for "
              << tnode_nodes.size() << " tnodes.\n";
}

// =============================================================================
// Step 4 – build CHANX/CHANY utilisation maps from router occupancy
// =============================================================================
void HeteroGraph::build_channel_util_map()
{
    const auto& device_ctx  = g_vpr_ctx.device();
    const auto& routing_ctx = g_vpr_ctx.routing();
    const auto& rr_graph    = device_ctx.rr_graph;

    grid_w = device_ctx.grid.width();
    grid_h = device_ctx.grid.height();

    if (grid_w == 0 || grid_h == 0) {
        grid_w = device_ctx.grid.width();
        grid_h = device_ctx.grid.height();
    }

    chanx_util_map.assign(grid_w, std::vector<float>(grid_h, 0.0f));
    chany_util_map.assign(grid_w, std::vector<float>(grid_h, 0.0f));

    // Record channel width for metadata output
    final_chan_width = (size_t)device_ctx.chan_width.x_max;

    // Iterate all RR nodes and record max utilisation per grid cell
    for (RRNodeId rr_id : rr_graph.nodes()) {
        e_rr_type type = rr_graph.node_type(rr_id);
        if (type != e_rr_type::CHANX && type != e_rr_type::CHANY) continue;

        int cap = rr_graph.node_capacity(rr_id);
        if (cap <= 0) continue;

        int occ   = routing_ctx.rr_node_route_inf[rr_id].occ();
        float util = (float)occ / (float)cap;

        size_t xlow = (size_t)rr_graph.node_xlow(rr_id);
        size_t ylow = (size_t)rr_graph.node_ylow(rr_id);

        if (xlow >= grid_w || ylow >= grid_h) continue;

        if (type == e_rr_type::CHANX) {
            chanx_util_map[xlow][ylow] = std::max(chanx_util_map[xlow][ylow], util);
        } else {
            chany_util_map[xlow][ylow] = std::max(chany_util_map[xlow][ylow], util);
        }
    }

    std::cout << "[TGNN] build_channel_util_map: grid " << grid_w << "x" << grid_h
              << ", chan_width=" << final_chan_width << "\n";
}

// =============================================================================
// Step 5 – enrich INTERCONNECT edge features and fill real net_delay
// =============================================================================
void HeteroGraph::enrich_edge_features(NetPinsMatrix<float> net_delay)
{
    build_channel_util_map();

    for (TEdge* edge : tnode_edges) {
        if (!edge->has_net_delay) continue;   // only INTERCONNECT

        // ---- 5a. Fill actual net delay ----
        edge->edge_delay = net_delay[edge->parent_net_id][edge->ipin] * 1.0e11f;

        // ---- 5b. Endpoint positions (prefer block_locs over rrnode coords) ----
        // Try tnode placement first; fall back to RR node coordinates
        auto fill_pos = [&](tatum::NodeId tid, int& ox, int& oy,
                            RRNode* fallback_rr) {
            if (get_block_loc(tid, ox, oy)) return;
            if (fallback_rr) {
                ox = (int)fallback_rr->xlow;
                oy = (int)fallback_rr->ylow;
            }
        };
        fill_pos(edge->src_tnode, edge->src_x, edge->src_y, edge->src_rrnode);
        fill_pos(edge->dst_tnode, edge->dst_x, edge->dst_y, edge->dst_rrnode);

        if (edge->src_x >= 0 && edge->dst_x >= 0)
            edge->manhattan_dist = (float)(std::abs(edge->dst_x - edge->src_x) +
                                           std::abs(edge->dst_y - edge->src_y));

        // ---- 5c. Net HPWL and fanout ----
        float hpwl = -1.0f;
        int   fo   = -1;
        if (get_net_hpwl_fanout(edge->parent_net_id, hpwl, fo)) {
            edge->net_hpwl   = hpwl;
            edge->net_fanout = fo;
        }

        // ---- 5d. Channel utilisation along the bounding box of the edge ----
        if (edge->src_x < 0 || edge->dst_x < 0) continue;

        size_t xlo = (size_t)std::max(0, std::min(edge->src_x, edge->dst_x));
        size_t xhi = (size_t)std::min((int)grid_w - 1, std::max(edge->src_x, edge->dst_x));
        size_t ylo = (size_t)std::max(0, std::min(edge->src_y, edge->dst_y));
        size_t yhi = (size_t)std::min((int)grid_h - 1, std::max(edge->src_y, edge->dst_y));

        float max_cx = 0.0f, max_cy = 0.0f;
        float sum_cx = 0.0f, sum_cy = 0.0f;
        size_t cnt = 0;

        for (size_t x = xlo; x <= xhi; ++x) {
            for (size_t y = ylo; y <= yhi; ++y) {
                float cx = chanx_util_map[x][y];
                float cy = chany_util_map[x][y];
                max_cx = std::max(max_cx, cx);
                max_cy = std::max(max_cy, cy);
                sum_cx += cx;
                sum_cy += cy;
                ++cnt;
            }
        }

        if (cnt > 0) {
            edge->path_max_chanx_util = max_cx;
            edge->path_max_chany_util = max_cy;
            edge->path_avg_chanx_util = sum_cx / (float)cnt;
            edge->path_avg_chany_util = sum_cy / (float)cnt;
        }
    }

    std::cout << "[TGNN] enrich_edge_features: done for "
              << tnode_edges.size() << " tedges.\n";
}

// =============================================================================
// Step 6 – write timing_graph.npz
// =============================================================================
void HeteroGraph::write_timing_graph_npz()
{
    // ---- Build index map ----
    std::unordered_map<tatum::NodeId, size_t> id_to_idx;
    id_to_idx.reserve(tnode_nodes.size());
    for (size_t i = 0; i < tnode_nodes.size(); ++i)
        id_to_idx[tnode_nodes[i]->id] = i;

    const size_t N = tnode_nodes.size();
    const size_t E = tnode_edges.size();

    // ---- Tnode arrays ----
    std::vector<size_t> tnode_type(N);
    std::vector<float>  tnode_rt_time(N);
    std::vector<float>  tnode_rt_required(N);
    std::vector<float>  tnode_rt_slack(N);
    std::vector<int>    tnode_valid_mask(N);
    std::vector<int>    tnode_x(N), tnode_y(N);
    std::vector<int>    tnode_fanin(N), tnode_fanout(N);
    std::vector<float>  tnode_net_hpwl(N);
    std::vector<int>    tnode_net_fanout(N);
    std::vector<float>  tnode_pl_arrival(N);
    std::vector<int>    tnode_pl_valid(N);
    std::vector<int>    tnode_topo_level(N);
    std::vector<int8_t> tnode_crit(N, 0);

    for (size_t i = 0; i < N; ++i) {
        const Tnode* n = tnode_nodes[i];
        tnode_type[i]        = static_cast<size_t>(n->type);
        tnode_rt_time[i]     = n->has_rt_arrival_time ? n->rt_arrival_time * 1.0e11f : -1.0f;
        tnode_rt_required[i] = n->has_rt_required     ? n->rt_required_time * 1.0e11f : -1.0f;
        tnode_rt_slack[i]    = (n->has_rt_arrival_time && n->has_rt_required)
                                   ? n->rt_slack * 1.0e11f : -1.0f;
        tnode_valid_mask[i]  = n->has_rt_arrival_time ? 1 : 0;
        tnode_x[i]           = n->blk_x;
        tnode_y[i]           = n->blk_y;
        tnode_fanin[i]       = n->fanin;
        tnode_fanout[i]      = n->fanout;
        tnode_net_hpwl[i]    = n->net_hpwl;
        tnode_net_fanout[i]  = n->net_fanout;
        tnode_pl_arrival[i]  = n->has_pl_arrival_time ? n->pl_arrival_time * 1.0e11f : -1.0f;
        tnode_pl_valid[i]    = n->has_pl_arrival_time ? 1 : 0;
        tnode_topo_level[i]  = n->topo_level;
        tnode_crit[i]        = (!tnode_on_critical_path.empty()) ? tnode_on_critical_path[i] : 0;
    }

    // ---- Tedge arrays ----
    std::vector<size_t> tedge_src(E), tedge_dst(E), tedge_type(E);
    std::vector<float>  tedge_delay(E);
    std::vector<int>    tedge_src_x(E), tedge_src_y(E);
    std::vector<int>    tedge_dst_x(E), tedge_dst_y(E);
    std::vector<float>  tedge_manhattan_dist(E);
    std::vector<float>  tedge_net_hpwl(E);
    std::vector<int>    tedge_net_fanout(E);
    std::vector<float>  tedge_path_max_chanx(E), tedge_path_max_chany(E);
    std::vector<float>  tedge_path_avg_chanx(E), tedge_path_avg_chany(E);

    for (size_t i = 0; i < E; ++i) {
        const TEdge* e = tnode_edges[i];
        tedge_src[i]             = id_to_idx.at(e->src_tnode);
        tedge_dst[i]             = id_to_idx.at(e->dst_tnode);
        tedge_type[i]            = static_cast<size_t>(e->edge_type);
        tedge_delay[i]           = e->edge_delay;
        tedge_src_x[i]           = e->src_x;
        tedge_src_y[i]           = e->src_y;
        tedge_dst_x[i]           = e->dst_x;
        tedge_dst_y[i]           = e->dst_y;
        tedge_manhattan_dist[i]  = e->manhattan_dist;
        tedge_net_hpwl[i]        = e->net_hpwl;
        tedge_net_fanout[i]      = e->net_fanout;
        tedge_path_max_chanx[i]  = e->path_max_chanx_util;
        tedge_path_max_chany[i]  = e->path_max_chany_util;
        tedge_path_avg_chanx[i]  = e->path_avg_chanx_util;
        tedge_path_avg_chany[i]  = e->path_avg_chany_util;
    }

    // ---- Top-K path arrays ----
    const int K_paths = (int)top_path_node_indices.size();
    int max_path_len = 0;
    for (const auto& p : top_path_node_indices)
        max_path_len = std::max(max_path_len, (int)p.size());

    // Flat [K_paths × max_path_len] with -1 padding
    std::vector<int> top_path_flat;
    std::vector<int> top_path_lengths(K_paths);
    if (K_paths > 0 && max_path_len > 0) {
        top_path_flat.assign((size_t)K_paths * (size_t)max_path_len, -1);
        for (int p = 0; p < K_paths; ++p) {
            top_path_lengths[p] = (int)top_path_node_indices[p].size();
            for (int j = 0; j < top_path_lengths[p]; ++j)
                top_path_flat[(size_t)p * (size_t)max_path_len + (size_t)j] =
                    top_path_node_indices[p][j];
        }
    }

    // ---- Global scalars ----
    std::vector<size_t> scalar_grid_w      = {grid_w};
    std::vector<size_t> scalar_grid_h      = {grid_h};
    std::vector<size_t> scalar_chan_width   = {final_chan_width};
    std::vector<float>  scalar_cpd          = {critical_path_delay * 1.0e11f};

    // ---- Write npz ----
    const std::string fname = "timing_graph.npz";

    // tnode
    cnpy::npz_save(fname, "tnode_type",           tnode_type.data(),        {N}, "w");
    cnpy::npz_save(fname, "tnode_rt_time",         tnode_rt_time.data(),     {N}, "a");
    cnpy::npz_save(fname, "tnode_rt_required",     tnode_rt_required.data(), {N}, "a");
    cnpy::npz_save(fname, "tnode_rt_slack",        tnode_rt_slack.data(),    {N}, "a");
    cnpy::npz_save(fname, "tnode_valid_mask",      tnode_valid_mask.data(),  {N}, "a");
    cnpy::npz_save(fname, "tnode_on_critical_path",tnode_crit.data(),        {N}, "a");
    cnpy::npz_save(fname, "tnode_x",               tnode_x.data(),           {N}, "a");
    cnpy::npz_save(fname, "tnode_y",               tnode_y.data(),           {N}, "a");
    cnpy::npz_save(fname, "tnode_fanin",           tnode_fanin.data(),       {N}, "a");
    cnpy::npz_save(fname, "tnode_fanout",          tnode_fanout.data(),      {N}, "a");
    cnpy::npz_save(fname, "tnode_net_hpwl",        tnode_net_hpwl.data(),    {N}, "a");
    cnpy::npz_save(fname, "tnode_net_fanout",      tnode_net_fanout.data(),  {N}, "a");
    cnpy::npz_save(fname, "tnode_pl_arrival",      tnode_pl_arrival.data(),  {N}, "a");
    cnpy::npz_save(fname, "tnode_pl_arrival_mask", tnode_pl_valid.data(),    {N}, "a");
    cnpy::npz_save(fname, "tnode_topo_level",      tnode_topo_level.data(),  {N}, "a");

    // tedge
    cnpy::npz_save(fname, "tedge_src",               tedge_src.data(),            {E}, "a");
    cnpy::npz_save(fname, "tedge_dst",               tedge_dst.data(),            {E}, "a");
    cnpy::npz_save(fname, "tedge_type",              tedge_type.data(),           {E}, "a");
    cnpy::npz_save(fname, "tedge_delay",             tedge_delay.data(),          {E}, "a");
    cnpy::npz_save(fname, "tedge_src_x",             tedge_src_x.data(),          {E}, "a");
    cnpy::npz_save(fname, "tedge_src_y",             tedge_src_y.data(),          {E}, "a");
    cnpy::npz_save(fname, "tedge_dst_x",             tedge_dst_x.data(),          {E}, "a");
    cnpy::npz_save(fname, "tedge_dst_y",             tedge_dst_y.data(),          {E}, "a");
    cnpy::npz_save(fname, "tedge_manhattan_dist",    tedge_manhattan_dist.data(), {E}, "a");
    cnpy::npz_save(fname, "tedge_net_hpwl",          tedge_net_hpwl.data(),       {E}, "a");
    cnpy::npz_save(fname, "tedge_net_fanout",        tedge_net_fanout.data(),     {E}, "a");
    cnpy::npz_save(fname, "tedge_path_max_chanx",    tedge_path_max_chanx.data(), {E}, "a");
    cnpy::npz_save(fname, "tedge_path_max_chany",    tedge_path_max_chany.data(), {E}, "a");
    cnpy::npz_save(fname, "tedge_path_avg_chanx",    tedge_path_avg_chanx.data(), {E}, "a");
    cnpy::npz_save(fname, "tedge_path_avg_chany",    tedge_path_avg_chany.data(), {E}, "a");

    // top-K paths
    if (K_paths > 0 && max_path_len > 0) {
        cnpy::npz_save(fname, "top_path_node_ids",
                       top_path_flat.data(),
                       {(size_t)K_paths, (size_t)max_path_len}, "a");
        cnpy::npz_save(fname, "top_path_delays",
                       top_path_delays.data(), {(size_t)K_paths}, "a");
        cnpy::npz_save(fname, "top_path_lengths",
                       top_path_lengths.data(), {(size_t)K_paths}, "a");
    }

    // global scalars
    cnpy::npz_save(fname, "grid_width",          scalar_grid_w.data(),    {1}, "a");
    cnpy::npz_save(fname, "grid_height",         scalar_grid_h.data(),    {1}, "a");
    cnpy::npz_save(fname, "chan_width",           scalar_chan_width.data(),{1}, "a");
    cnpy::npz_save(fname, "critical_path_delay", scalar_cpd.data(),       {1}, "a");

    std::cout << "[TGNN] write_timing_graph_npz: "
              << N << " tnodes, " << E << " tedges, "
              << K_paths << " critical paths -> " << fname << "\n";

    // Optional: write BLIF name annotations for debug
    std::ofstream ofs("tnode_blif.txt");
    for (const auto& [tid, name] : tnode_blif_name)
        ofs << static_cast<size_t>(tid) << '\t' << name << '\n';
}

} // namespace tgnn