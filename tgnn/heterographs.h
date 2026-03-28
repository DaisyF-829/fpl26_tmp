#ifndef HETEROGRAPHS_H
#define HETEROGRAPHS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "atom_delay_calc.h"
#include "concrete_timing_info.h"
#include "vpr_net_pins_matrix.h"

/*
 * Heterogeneous timing graph for TGNN feature extraction.
 *
 * Public pipeline:
 *   build_from_timing_graph_prepl()   — tnodes only (no tedges)
 *   build_rr_node_map()               — tile map + extract_used_rrnodes + prune + remap
 *   extract_arrival_time_aftroute()   — routed STA
 *   extract_arrival_time_preroute()     — placement STA → pl_arrival_time
 *   extract_top_k_paths()
 *   extract_node_features()
 *   enrich_edge_features()
 *   write_timing_graph_npz()
 */

namespace tgnn {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const noexcept {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct RRNode {
    RRNodeId id;
    size_t   node_type;
    size_t   xlow, xhigh, ylow, yhigh;
    float    r, c;
    size_t   usage_count = 0;
};

struct RREdge {
    RRNodeId src_rr{};
    RRNodeId dst_rr{};
    float    switch_delay = 0.f;
    float    r            = 0.f;
    float    c_in         = 0.f;
    float    c_out        = 0.f;
};

struct Tile {
    size_t              id                 = 0;
    size_t              x                  = 0;
    size_t              y                  = 0;
    std::vector<RRNode*> rr_nodes;
    std::vector<RREdge*> rr_edges;
    size_t              usage_count        = 0;
    size_t              source_sink_count  = 0;
};

struct RRTileEdge {
    RRNodeId rrnode_id{};
    size_t   tile_id = 0;
};

struct TileEdge {
    size_t src_tile   = 0;
    size_t dst_tile   = 0;
    size_t num_edges  = 0;
};

struct Tnode {
    tatum::NodeId   id;
    tatum::NodeType type;

    float rt_arrival_time     = 0.0f;
    bool  has_rt_arrival_time = false;
    float rt_required_time    = 0.0f;
    bool  has_rt_required     = false;
    float rt_slack            = 0.0f;

    float pl_arrival_time     = 0.0f;
    bool  has_pl_arrival_time = false;

    int blk_x = -1;
    int blk_y = -1;

    int fanin  = 0;
    int fanout = 0;

    float net_hpwl   = -1.0f;
    int   net_fanout = -1;

    int topo_level = -1;
};

struct TEdge {
    tatum::NodeId   src_tnode;
    tatum::NodeId   dst_tnode;
    tatum::EdgeType edge_type;
    float           edge_delay = 0.0f;

    bool        has_net_delay = false;
    ParentNetId parent_net_id = {};
    size_t      ipin          = 0;

    RRNode* src_rrnode = nullptr;
    RRNode* dst_rrnode = nullptr;

    int   src_x = -1, src_y = -1;
    int   dst_x = -1, dst_y = -1;
    float manhattan_dist = -1.0f;

    float net_hpwl   = -1.0f;
    int   net_fanout = -1;

    float path_max_chanx_util = -1.0f;
    float path_max_chany_util = -1.0f;
    float path_avg_chanx_util = -1.0f;
    float path_avg_chany_util = -1.0f;
};

class HeteroGraph {
  public:
    static HeteroGraph& get_instance() {
        static HeteroGraph instance;
        return instance;
    }
    HeteroGraph(const HeteroGraph&)            = delete;
    HeteroGraph& operator=(const HeteroGraph&) = delete;

    void build_from_timing_graph_prepl();
    void build_rr_node_map();

    void extract_arrival_time_aftroute(const SetupTimingInfo& timing_info);
    void extract_arrival_time_preroute(const SetupTimingInfo& timing_info);

    void extract_top_k_paths(const SetupTimingInfo& timing_info, int K = 20);

    void extract_node_features();

    void enrich_edge_features(NetPinsMatrix<float> net_delay);

    void write_timing_graph_npz();

  private:
    HeteroGraph() = default;

    void free_rr_tile_artifacts();

    std::vector<Tnode*> tnode_nodes;
    std::vector<TEdge*> tnode_edges;

    std::unordered_map<tatum::NodeId, Tnode*> tnode_ptr_map;
    std::unordered_map<tatum::NodeId, size_t> tnode_id_map;

    std::unordered_map<tatum::NodeId, std::string> tnode_blif_name;
    std::unordered_map<tatum::NodeId, bool>        tnode_has_real_net;
    const std::string no_pin_placeholder = "[VIRTUAL]";
    const std::string no_net_placeholder = "[NO_NET]";

    std::unordered_map<RRNodeId, RRNode*> used_rrnodes;

    std::unordered_map<std::pair<size_t, size_t>, size_t, pair_hash> tile_coord_to_id_;
    std::vector<Tile*>                                               tiles_by_id_;
    std::multimap<tatum::NodeId, RRNodeId>                            tnode_rr_edges_;

    std::vector<RRNode*>           pruned_rr_nodes_;
    std::vector<RREdge*>          pruned_rr_edges_;
    std::unordered_map<RRNodeId, size_t> id_remap_;

    std::vector<RRTileEdge*> rr_tile_edges_;
    std::vector<TileEdge*>   tile_edges_;
    std::unordered_map<RRNodeId, RRTileEdge*> rr_id_to_rrt_edge_map_;

    size_t grid_w = 0;
    size_t grid_h = 0;

    std::vector<std::vector<int>> top_path_node_indices;
    std::vector<float>            top_path_delays;
    std::vector<int8_t>           tnode_on_critical_path;

    std::vector<std::pair<tatum::NodeId, size_t>> pruned_tnode_rr_edges_;

    float critical_path_delay = 0.0f;

    bool get_block_loc(tatum::NodeId node_id, int& x, int& y) const;
    bool get_net_hpwl_fanout(ParentNetId net_id, float& hpwl, int& fanout) const;

    void extract_used_rrnodes(const Netlist<>& netlist);

    void initialize_tile_map();
    void prune_rr_graph();
    void update_pruned_tnode_rr_edges();
};

} // namespace tgnn

#endif
