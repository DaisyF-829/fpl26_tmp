\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
%\pagestyle{plain} % removes running headers
\usepackage{fancyhdr}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{threeparttable}
% \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{algpseudocode}
\usepackage{bbding}
\usepackage{amsmath}
\usepackage{graphicx}



\pagestyle{empty}
\setlength{\textfloatsep}{1pt} 
\setlength{\floatsep}{1pt}
\setlength{\abovecaptionskip}{0.2cm}
% 减小公式与上文的间距
\setlength\abovedisplayskip{0pt}
% 减小公式与下文的间距
\setlength\belowdisplayskip{0pt}
% \addtolength{\topmargin}{-0.47in} 
% \addtolength{\textheight}{0.42in} 
% %% \BibTeX command to typeset BibTeX logo in the docs
\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}


\begin{document}

\title{PASTE: A \underline{P}hysical-\underline{A}ware \underline{S}urrogate for FPGA Pre-Routing \underline{T}iming \underline{E}stimation、、g}
\maketitle

\begin{abstract}
% Traditional FPGA retiming optimization is done during logic synthesis without accounting for physical timing constraints, resulting in suboptimal performance when routing delays are considered. Although the full place-and-route (P\&R) flow provides accurate timing, its high computational cost makes it impractical for iterative optimization.
% This paper proposes a novel physical-aware retiming approach that incorporates a pre-routing timing predictor.
% Specifically, we partition the Routing Resource Graph (RRG) into intra-tile subgraphs and employ a pre-trained Graph AutoEncoder to learn both local node embeddings and global tile embeddings. The learned embeddings are integrated with features extracted from the timing graph and the physical placement of nets to predict net delays, arrival times, and the critical paths.
% Leveraging the predictions, we evaluate the retiming moves along them and guide ABC's heuristic to target the true post-routing bottlenecks, optimizing timing without requiring repeated P\&R runs.
% Our predictor, incorporated into VTR9, is validated on the MCNC and VTR benchmarks, achieving an $R^2$ of 0.90 for arrival time prediction of the endpoints and up to 56x speedup for large circuits. Experiments show an average 8\% improvement in maximum frequency over ABC’s logic-only retiming. This physical-aware retiming bridges the gap between logic and physical optimization, improving timing in modern FPGA design flows.


Accurate pre-routing timing estimation is a fundamental challenge in FPGA design, since the final path delay is jointly determined by logic structure, placement, routing-resource constraints, and congestion. Existing early-stage methods often rely on coarse proxies or simplified delay models, which limits their usefulness for node-level timing analysis and critical-path identification before detailed routing.
In this paper, we present PASTE, a physical-aware surrogate for FPGA pre-routing timing estimation. PASTE first extracts physically informative features from the timing graph of a placed design, including placement-derived geometry, routing-related attributes, and congestion-aware signals, so as to capture routing-induced timing variation beyond pure logic topology. Based on these features, we further propose a novel Timing Propagation Network (TPN), a directed edge-aware graph neural network, to predict node-level normalized arrival times and graph-level critical path delay.
Implemented in VTR, PASTE achieves an $R^2$ of 0.98 and a Spearman correlation of 0.98 on the test set, while requiring only millisecond-level inference time, which is substantially smaller than the runtime of actual routing and static timing analysis. As a downstream validation case, we further integrate PASTE into a predictor-guided retiming flow, which improves $F_{\max}$ by 8\% on average over logic-only ABC retiming. These results demonstrate that physical-aware graph-based timing estimation can serve as an effective surrogate for early-stage timing analysis and timing-driven FPGA optimization.

\end{abstract}


\begin{IEEEkeywords}
GNN, STA, Timing Prediction
\end{IEEEkeywords}




\section{Introduction}

FPGA-based systems have become central to modern hardware acceleration, offering high flexibility and performance. Retiming is a fundamental sequential optimization technique that reduces critical-path logic depth by relocating registers across combinational logic blocks~\cite{ref:retime}.

Despite its effectiveness in early design stages, traditional retiming is performed without knowledge of physical timing constraints, and thus relies on overly simplified, logic-only delay models~\cite{ref:abcret}. Moreover, the routing delays can account for more than half of the total critical-path delay~\cite{ref:route_delay} in modern FPGAs. Consequently, logic-only retiming often misidentifies the true bottlenecks and leads to suboptimal $F_\text{max}$ improvements. These limitations motivate the need for physical-aware retiming methods that integrate logic synthesis with realistic timing behaviors induced by placement and routing.

Several approaches have emerged to address this challenge. One strategy involves performing complete placement and routing iterations during retiming optimization~\cite{ref_iterret}. While this methodology delivers highly accurate timing information, it imposes prohibitive computational overhead, with individual P\&R iterations requiring hours to days for large-scale FPGA designs. This computational burden renders iterative retiming optimization impractical, particularly in complex systems requiring multiple design iterations.
Alternative research efforts have explored latch-based designs~\cite{ref:latret} and physical-aware retiming methodologies that incorporate post-placement timing information for enhanced optimization accuracy. However, these approaches often struggle to capture the full complexity of routing effects~\cite{b3}, either limiting analysis to local routing or employing oversimplified delay models such as those used in retiming-based timing analysis (RTA)~\cite{ref:rta}, which introduce significant prediction inaccuracies through simplified timing graph representations. 

\begin{table}[t]
\centering
\caption{Related Works in Timing Prediction}
\label{tab:related}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{c|cccc}
\toprule
\textbf{Reference} &
\textbf{Domain} &
\textbf{Granularity} &
\begin{tabular}[c]{@{}c@{}}\textbf{Design-}\\\textbf{Aware}\end{tabular} &
\textbf{Metric} \\
\midrule

\cite{WCPNet,wirelength} & FPGA & Circuit-level & \CheckmarkBold & Wirelength \\
\cite{bbcong,conghls,fGREP} & FPGA & Region-level & \CheckmarkBold & Congestion \\
\cite{dl_routability,routenet,sta_rout_mux} & FPGA & Circuit-level & \CheckmarkBold & Routability \\
\cite{hlspd} & FPGA & Op-level & \CheckmarkBold & Operation delay \\
\cite{fpga_ml_delay} & FPGA & Net-level & \XSolidBrush & Net delay \\
\cite{air} & FPGA & Net-level & \XSolidBrush & Routing cost \\
\cite{timinggcn,letime,gattiming,preroutegnn} & ASIC & Cell-level & \CheckmarkBold & \textbf{Arrival Time} \\
Ours & FPGA & Register-level & \CheckmarkBold & \textbf{Arrival Time} \\
\bottomrule
\end{tabular}%
}
\end{table}



The emergence of machine learning has brought new opportunities.
Recent studies demonstrate that graph neural networks (GNNs) can capture long-range timing dependencies in ASIC design flows, enabling accurate pre-routing delay estimation~\cite{timinggcn,preroutegnn}.
However, applying these methods to FPGAs remains challenging. In FPGA designs, net delay is dominated by architecture‑specific routing fabrics, which impose strict and non‑customizable routing constraints. These constraints, including limited path options, fixed channel widths, and delay variations due to congestion, complicate pre-routing timing modeling.
As shown in Table~\ref{tab:related}, prior FPGA timing-prediction methods either predict indirect, coarse-grained metrics or provide architecture-only delay estimates that do not incorporate design-specific structure or timing dependencies. 
This highlights the need for FPGA-specific models that directly estimate node-level delays across real designs.


In this paper, we propose a novel physical-aware retiming approach accelerated by a pre-routing timing predictor, as shown in Fig.~\ref{fig:framework}. The main contributions are as follows:

\begin{itemize}
    \item \textbf{First FPGA Pre-routing Node-level Arrival Time (AT) Predictor}: The predictor incorporates placement, topology, and congestion information.
    \item \textbf{First Use of Graph AutoEncoder (GAE) for FPGA RRG Feature Enhancement}: We model intra-tile RRG and apply GAE with an edge-aware network layer to enhance node embeddings and improve feature learning.
    \item \textbf{Physical-aware Retiming Algorithm}: We propose a retiming algorithm that integrates predicted delays, optimizing critical paths while considering physical constraints.
    \item \textbf{Experimental Validation}: Comprehensive experiments demonstrate that our approach improves $F_\text{max}$ by 8\% compared to traditional logic-only retiming methods.
\end{itemize}


\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figure/framwork.txt.png}
    \caption{Traditional physical-aware retiming~\cite{ref_iterret} and the proposed framework.}
    \label{fig:framework}
    \vspace{4pt}
\end{figure}

\section{Background and Preliminaries}
% \subsection{Retiming}
% Retiming is a classic sequential circuit optimization technique used to improve the clock frequency of a design by repositioning registers across combinational logic blocks without altering the circuit's functional behavior~\cite{ref:retime}. The primary objective of retiming is Clock Period Minimization, which focuses on reducing the maximum delay of any combinational path between two consecutive registers. By strategically moving registers, this optimization directly minimizes the circuit's minimum achievable clock period, thereby increasing its maximum operating frequency.

% To formalize this process, a circuit is modeled as a directed graph, \(G=(V,E)\), where each vertex \(v \in V\) represents a combinational logic gate, and each directed edge \(e=(u,v) \in E\) represents a signal connection from gate $u$ to gate $v$ . Each vertex \(v\) is assigned a weight \(d(v)\) corresponding to its propagation delay. Edges are assigned weights,\(w(e)\), representing the number of registers on that connection. The core of retiming involves assigning an integer lag, \(r(v)\), to each vertex. This lag signifies moving \(r(v)\) registers from the output of gate $v$ to its inputs. Consequently, the register count on an edge \(e=(u,v)\) is updated to \(w'(e)= w ( e ) + r ( v ) - r ( u )\). For a valid retiming, all \(w'(e)\) must remain non-negative. The optimization problem is then to find a lag assignment \(r\) that minimizes the clock period or register count under the relevant constraints.

\subsection{FPGA Routing and Static Timing Analysis}

FPGA Static Timing Analysis (STA) comprises clock-to-Q delay, combinational delay, and net delay, as illustrated in Fig.~\ref{fig:rrg}. 
Among these components, the first two are determined by cell characteristics, whereas net delay depends on the routing path and therefore can only be accurately determined after detailed routing.

Unlike ASICs, where a net can be approximated as a single geometric connection with predictable delay behavior, an FPGA net is implemented by selecting a path through the fixed routing architecture. The architecture is typically modeled as a RRG, shown in Fig.~\ref{fig:rrg}, where nodes correspond to wires or pins and edges denote possible programmable connections. Routing therefore involves choosing a legal path for each net that satisfies
architectural constraints and avoids resource conflicts.

In practice, the delay of an FPGA net is influenced by the switch-box topology, local congestion, and the distance between the source and the sink. These effects result in strongly architecture- and congestion-dependent delay pattern, which cannot be captured by location-based models.  
This highlights the need to model the RRG topology explicitly for accurate pre-routing predictions.

% \subsection{GNN}

% GNNs are models designed for graph-structured data~\cite{gnn}. Through message passing, GNNs aggregate features from neighboring nodes and iteratively update node representations, capturing complex dependencies within the graph. By stacking multiple layers, the model gains a broader receptive field, enabling it to capture dependencies across larger portions of the graph. The message-passing operation can be expressed as follows:
% \begin{equation}
% \mathbf{h}_v^{(k+1)} = \sigma\left( W \cdot \text{AGG}\left( \{ \mathbf{h}_u^{(k)} : u \in \mathcal{N}(v) \} \right) + b \right),
% \end{equation}
% where \(\mathbf{h}_v^{(k)}\) represents the feature vector of node \(v\) at layer \(k\), and \(\mathcal{N}(v)\) denotes the set of neighbors of node \(v\). This aggregation process allows the model to progressively incorporate contextual information from larger neighborhoods, expanding its ability to capture global dependencies.

% In FPGA design, the bottleneck in path-level timing prediction lies in the accurate estimation of net delay. The paths or trajectories of the nets are established on the basis of the FPGA routing architecture, which is described by the RRG. In the RRG, nodes represent routing resources, while edges indicate that two resources can be connected via switches. GNNs can effectively capture the topological structure of the routing architecture and the delay characteristics of the routing resources, making them ideal for learning embeddings that is crucial for net delay prediction.



% \section{GAE-enhanced Timing Predictor}
% \section{Overall flow}
% Our proposed framework is illustrated in Fig.~\ref{fig:framework}, which contrasts it with a previous physical-aware retiming optimization flow~\cite{ref_iterret}. 
% The entire process is integrated into the Verilog-to-Routing (VTR)~\cite{vtr9}, a comprehensive academic CAD flow that translates Verilog designs into FPGA configurations. The flow begins with logic synthesis, where the input design is processed by tools such as ABC~\cite{abc} to generate a technology-mapped netlist.
% As depicted on the left side of Figure~\ref{fig:framework}, the conventional physical-aware retiming in~\cite{ref_iterret} suffers from a significant performance bottleneck. To obtain accurate timing information for the physical-aware retiming engine, the design must pass through the complete P\&R flow. Within the VTR toolchain, this task is handled by the Versatile Place and Route (VPR) engine. The bottleneck arises because the entire VPR execution (P\&R, STA) is enclosed within a slow, iterative loop. Consequently, each retiming decision necessitates a full P\&R run, making the optimization process prohibitively time-consuming for large designs.

% Our framework, shown on the right, fundamentally accelerates this process by decoupling the retiming loop from the expensive routing stage. After an initial VPR Pack \& Place, instead of running the full router, we leverage our GAE-enhanced Timing predictor. This predictor takes the placed netlist, the timing graph, and the routing resource graph as input, estimates the arrival time of each node, and outputs the predicted longest 10 paths. The predicted critical paths are then fed to the Physical-aware retiming engine, forming a new, lightweight optimization loop (indicated in green). This loop, comprising only placement, prediction, and retiming, can iterate faster than the conventional approach. By replacing the costly physical routing step with a fast and accurate pre-routing timing prediction, our framework effectively bridges the gap between logical optimization and physical reality without incurring the prohibitive runtime overhead.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{figure/rrg.png}
    \caption{Mapping physical routing elements to RRG and the timing graph.}
    \label{fig:rrg}
    \vspace{4pt}
\end{figure}

% \section{Framework Overview}

% Our proposed framework is integrated into the Verilog-to-Routing (VTR) framework~\cite{vtr9}, a comprehensive academic CAD flow that translates Verilog designs into FPGA configurations.

% As shown in Fig.~\ref{fig:framework}, traditional physical-aware retiming methods~\cite{ref_iterret} incur significant computational overhead due to the need for a complete routing process at each retiming decision. In contrast, our method accelerates optimization by using a GAE-enhanced Timing predictor. This predictor takes the placed netlist, the timing graph, and routing resource graph as inputs, estimates the arrival time of each node, and outputs the predicted longest 10 paths.
% These 10 paths are then used to identify the most critical timing paths in the circuit, which are subsequently targeted for retiming adjustments. By relocating the registers in these paths, we can effectively reduce delay and improve the circuit’s overall timing performance.

% By replacing routing with fast timing prediction, our framework reduces overhead and bridges the gap between logical optimization and physical reality.


% \subsection{GAE-Enhanced Timing Predictor}



% The critical path is determined by the timing graph structure, logic delays, and net delays. Among these, net delay is influenced by the routing and Static Timing Analysis (STA) stages, which depend on factors such as the physical distance between pins, routing architecture, and congestion levels. To comprehensively address these factors, we propose a GAE-enhanced predictor that improves feature learning through the aggregation of both local and global dependencies.

% In the timing graph, net delay is represented as the delay along an interconnect edge, while in the RRG, it is represented as the path between a source–sink (SS) pair, where both nodes are virtual node types. As illustrated in Fig.~\ref{fig:gnnpd}, our feature extraction and prediction processes are based on this representation. Specifically, our inference workflow consists of three major stages:

% \subsubsection{Feature Engineering}
% Feature engineering in our framework involves three key aspects. 

%  \textbf{\underline{Interconnect Features in Timing Graph}:} In the Timing graph, each interconnect edge represents an SS pair. The in-degree and out-degree of nodes impact local connectivity complexity. The number of source and sink nodes, as well as the average and maximum node degrees, affect global routing behavior. Additionally, graph density, the ratio of edges to the maximum possible edges, reflects overall connectivity in the graph, leading to congestion, detours, and increased net delay.
 

%  \textbf{\underline{SS Pair Features}:} This category focuses on the macroscopic characteristics of the source and sink nodes. It includes the relative physical location between the source and sink, which directly influences the physical transmission distance of the net, thereby significantly impacting the net delay. Additionally, we quantify the number of valid nodes within the tile of the source or sink, which contributes to local congestion.


%  \textbf{\underline{Tile-level RRG Features}:} For a given RRG, we begin by pruning the graph based on the placement mapping results. We retain only those nodes that can be connected to valid source or sink nodes, removing redundant or unreachable nodes. The second step is intra-tile RRG construction. Specifically, we categorize all nodes based on their coordinates, assigning each node to its corresponding tile. For wires that span multiple tiles, we ensure that each node in the span is included in the respective tiles.
 
% Since source and sink nodes are virtual and lack delay features, we introduce a GAE to learn delay and topological features. The GAE model is pre-trained by minimizing the reconstruction error, ensuring that the learned embeddings preserve both local node features and global structural information. 
% Our encoder learns hierarchical dependencies using a three-layer GNN, which helps capture both local and global interactions for more accurate predictions.
% During inference, only the encoder generates node and graph embeddings, with the latter obtained by averaging the node embeddings of all nodes in the graph.

% For each SS pair, we extract the bounding box that encloses both the source and sink. The features of each SS pair include node embeddings from the pre-trained GAE and the average graph embeddings of the graphs within the bounding box. Additionally, we count the number of times each tile appears in different bounding boxes. A higher frequency indicates greater overlap, suggesting potential congestion in that region.

% The SS pair features are the concatenation of these three types of features, providing both local and global perspectives on the interconnect characteristics.

% \subsubsection{XGBoost-based Net Delay Predictor}
% The SS feature vectors are fed into an XGBoost regressor for net delay prediction. XGBoost is an advanced gradient boosting framework that constructs an ensemble of decision trees by sequentially fitting new trees to the residuals of previous ones, thereby capturing complex nonlinear relationships in the data. During training, the model is fitted using datasets derived from VPR routing and STA results.

% \subsubsection{Arrival Time and Critical Path Prediction}
% In the timing graph, delays are represented on the edges and are categorized into three types: \textbf{Clock-to-Q delay}, \textbf{Combinational logic delay}, and \textbf{Net delay}. The first two types of delays are obtained directly from the architecture file, while the \textbf{Net delay} is predicted by an \textbf{XGBoost-based model}. The arrival time (AT) of each node is computed through a forward traversal of the timing graph. The arrival time of each node can be expressed as follows:
% \begin{equation}
% AT(v_t) = \max_{u_t \in \text{pred}(v_t)} \left[ AT(u_t) + \text{delay}(u_t \rightarrow v_t) \right],
% \end{equation}
% where \( \text{pred}(v_t) \) denotes the set of immediate predecessors of node \( v_t \), and \( \text{delay}(u_t \rightarrow v_t) \) is the delay on the edge from node \( u_t \) to node \( v_t \).
% Based on the nodes with the maximum arrival times, we trace back the longest delay path, which will be used as input for the subsequent retiming.


\section{Physical-Aware Timing Predictor}
PASTE relies on physically informative node and edge features extracted from the placed timing graph, so that the subsequent propagation network can model both logical timing dependencies and implementation-aware delay variation.

\subsection{RRG Pruning}
\label{sec:rrg_pruning}

The original routing resource graph (RRG) contains all physical routing
resources and their interconnections. However, for a given design, only
a subset of source and sink RRG nodes are associated with timing nodes.
We denote these mapped nodes by
$V_{\mathrm{rr,src}}^{\mathrm{valid}}$ and
$V_{\mathrm{rr,sink}}^{\mathrm{valid}}$, respectively. Since the
subsequent routing-related feature extraction only depends on resources
that are reachable from valid sources and can further connect to valid
sinks, we prune the RRG before computing density statistics.

Specifically, each timing node is first mapped to its corresponding RRG
source or sink node through the placement and net connectivity
information provided by VTR. We then perform a forward breadth-first
search (BFS) from $V_{\mathrm{rr,src}}^{\mathrm{valid}}$ and a backward
BFS from $V_{\mathrm{rr,sink}}^{\mathrm{valid}}$. The valid RRG node set
is defined as the intersection of the forward-reachable and
backward-reachable nodes, and the valid edge set is induced by the
resulting valid node set. The complete procedure is summarized in
Algorithm~\ref{alg:rrg_pruning}. Since both BFS traversals run in
$O(|V|+|E|)$ time, the pruning overhead is modest even for large RRGs.
All routing-density features in the subsequent sections are computed on
the pruned graph using
$V_{\mathrm{rr}}^{\mathrm{valid}}$ and
$E_{\mathrm{rr}}^{\mathrm{valid}}$.

\begin{algorithm}[t]
\caption{RRG Pruning via Breadth-First Search}
\label{alg:rrg_pruning}
\small
\begin{algorithmic}[1]
\REQUIRE Timing-node set $V_t$, RRG node set $V_{\mathrm{rr}}$, RRG edge set $E_{\mathrm{rr}}$
\ENSURE Valid RRG node set $V_{\mathrm{rr}}^{\mathrm{valid}}$, valid RRG edge set $E_{\mathrm{rr}}^{\mathrm{valid}}$

\STATE $(V_{\mathrm{rr,src}}^{\mathrm{valid}},\,V_{\mathrm{rr,sink}}^{\mathrm{valid}})\gets \mathrm{MAP}(V_t)$
\STATE $S_{\mathrm{valid}} \gets V_{\mathrm{rr,src}}^{\mathrm{valid}}$
\STATE $T_{\mathrm{valid}} \gets V_{\mathrm{rr,sink}}^{\mathrm{valid}}$
\STATE $S_{\mathrm{valid}} \gets S_{\mathrm{valid}} \cup \mathrm{BFS\mbox{-}F}(S_{\mathrm{valid}}, V_{\mathrm{rr}}, E_{\mathrm{rr}})$
\STATE $T_{\mathrm{valid}} \gets T_{\mathrm{valid}} \cup \mathrm{BFS\mbox{-}B}(T_{\mathrm{valid}}, V_{\mathrm{rr}}, E_{\mathrm{rr}})$
\STATE $V_{\mathrm{rr}}^{\mathrm{valid}} \gets S_{\mathrm{valid}} \cap T_{\mathrm{valid}}$
\STATE $E_{\mathrm{rr}}^{\mathrm{valid}} \gets \{(v_{\mathrm{src}},v_{\mathrm{dst}})\in E_{\mathrm{rr}} \mid v_{\mathrm{src}},v_{\mathrm{dst}}\in V_{\mathrm{rr}}^{\mathrm{valid}}\}$
\RETURN $V_{\mathrm{rr}}^{\mathrm{valid}}, E_{\mathrm{rr}}^{\mathrm{valid}}$
\end{algorithmic}
\end{algorithm}

\subsection{Feature Engineering}

For each timing graph, we construct node and edge features to describe
topological structure, placement geometry, timing priors, and
routing-related context. Common scalar attributes are normalized before
being fed into the model, while categorical attributes are encoded by
one-hot representations. All density-related statistics are computed on
the pruned RRG introduced in Sec.~\ref{alg:rrg_pruning}, using the valid
node set $V_{\mathrm{rr}}^{\mathrm{valid}}$ and valid edge set
$E_{\mathrm{rr}}^{\mathrm{valid}}$. The complete node and edge features
are summarized in Table~\ref{tab:node_feat_timing} and
Table~\ref{tab:edge_feat_timing}.

\begin{table*}[t]
\centering
\caption{Node features used in the timing graph.}
\label{tab:node_feat_timing}
\footnotesize
\begin{tabular}{llll}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type/Range} & \textbf{Description} \\
\midrule
\multirow{2}{*}{Placement}
& x-coordinate
& float / normalized
& horizontal location; available for placed node types only \\
& y-coordinate
& float / normalized
& vertical location; available for placed node types only \\
\midrule
\multirow{4}{*}{Topology}
& logic level
& integer
& longest topological distance in the combinational subgraph \\
& global longest logic level
& integer / broadcast
& maximum logic level of the graph, shared by all nodes \\
& node type
& categorical / one-hot
& node semantics \\
& fanin / fanout
& integer
& local connectivity \\
\midrule
\multirow{1}{*}{Timing Prior}
& post-placement arrival-time prior
& float / normalized
& placement-stage timing estimate from VTR \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Edge features used in the timing graph.}
\label{tab:edge_feat_timing}
\footnotesize
\begin{tabular}{llll}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type/Range} & \textbf{Description} \\
\midrule
\multirow{2}{*}{Geometry}
& HPWL
& float / normalized
& source--sink bounding-box wirelength \\
& Manhattan distance
& float / normalized
& source--sink geometric span \\
\midrule
\multirow{2}{*}{Bounding-box Density}
& node-density max / avg
& float / normalized
& valid source/sink density over covered tiles \\
& edge-usage-density max / avg
& float / normalized
& valid routing-demand density over covered tiles \\
\bottomrule
\end{tabular}
\end{table*}

For each logic node $v$, we compute its logic level as the longest
topological distance from a source logic node in the combinational
subgraph:
\begin{equation}
l_v=\max_{p\in\mathcal{P}(v)} |p|,
\end{equation}
where $\mathcal{P}(v)$ denotes the set of topological paths ending at
$v$. We further compute the maximum logic level of the whole graph,
\begin{equation}
l_{\max}=\max_{v\in\mathcal{V}} l_v,
\end{equation}
and broadcast it to all nodes as a global topological feature.

In addition, we extract a post-placement arrival-time prior from the
timing information generated by VTR after placement. This prior is
mainly derived from combinational delay propagation and is normalized by
the maximum post-placement arrival time of the same design:
\begin{equation}
\tilde{t}^{\mathrm{pl}}_v
=
\frac{t^{\mathrm{pl}}_v}{\max_{u\in\mathcal{V}} t^{\mathrm{pl}}_u}.
\end{equation}

For each edge $(u,v)$, let $\mathcal{B}_{uv}$ denote the set of tiles
covered by the source--sink bounding box. We first map the placed timing
nodes to their corresponding RRG nodes and partition them according to
tile locations. For each tile $b\in\mathcal{B}_{uv}$, the node density
is defined as
\begin{equation}
\rho^{\mathrm{node}}_b
=
\frac{N^{\mathrm{valid}}_b}{N^{\mathrm{all}}_b},
\end{equation}
where $N^{\mathrm{valid}}_b$ is the number of valid source/sink RRG
nodes in tile $b$, and $N^{\mathrm{all}}_b$ is the total number of
source/sink RRG nodes in that tile. The maximum and average values
within the bounding box are then used as edge features:
\begin{equation}
\rho^{\mathrm{node}}_{uv,\max}
=
\max_{b\in\mathcal{B}_{uv}} \rho^{\mathrm{node}}_b,
\end{equation}
\begin{equation}
\rho^{\mathrm{node}}_{uv,\mathrm{avg}}
=
\frac{1}{|\mathcal{B}_{uv}|}\sum_{b\in\mathcal{B}_{uv}}
\rho^{\mathrm{node}}_b.
\end{equation}

Similarly, each tile accumulates one count whenever it is covered by the
bounding box of any net. Let $C_b$ denote the accumulated bounding-box
count on tile $b$, and let $E^{\mathrm{valid}}_b$ denote the number of
valid routing edges in that tile. The edge-usage density is defined as
\begin{equation}
\rho^{\mathrm{edge}}_b
=
\frac{C_b}{E^{\mathrm{valid}}_b},
\end{equation}
and its maximum and average values over $\mathcal{B}_{uv}$ are used as
edge features:
\begin{equation}
\rho^{\mathrm{edge}}_{uv,\max}
=
\max_{b\in\mathcal{B}_{uv}} \rho^{\mathrm{edge}}_b,
\end{equation}
\begin{equation}
\rho^{\mathrm{edge}}_{uv,\mathrm{avg}}
=
\frac{1}{|\mathcal{B}_{uv}|}\sum_{b\in\mathcal{B}_{uv}}
\rho^{\mathrm{edge}}_b.
\end{equation}

\subsection{Timing Propagation Network}

Given a timing graph $\mathcal{G}=\left(\mathcal{V}, \left\{\mathcal{E}^{(k)}\right\}_{k=0}^{3}\right)$,
where $\mathcal{V}$ denotes the set of timing nodes and $\mathcal{E}^{(k)}$
denotes the directed edge set of relation type $k$.Each node $v\in\mathcal{V}$ is
associated with a node feature vector $\mathbf{x}_v$, and each directed
edge $(u,v)\in\mathcal{E}^{(k)}$ is associated with an edge feature
vector $\mathbf{a}_{uv}^{(k)}$. The goal is to
predict the normalized arrival time for each node and the critical path
delay (CPD) of the whole graph.

To process this graph, we use a \emph{Timing Propagation Network} (TPN),
which performs directed message passing over the four edge types and
produces both node-level and graph-level outputs. Specifically, the
network predicts the normalized arrival time of each node and the CPD of
the entire timing graph.

\subsubsection{Input encoding}
We first project raw node and edge features into a shared latent space.
For each node $v$ and each edge $(u,v)\in\mathcal{E}^{(k)}$, the encoded
representations are given by the equations below.
\begin{equation}
\mathbf{h}_v^{(0)}=\mathrm{MLP}_{\mathrm{node}}(\mathbf{x}_v)\in\mathbb{R}^{d},
\end{equation}
\begin{equation}
\mathbf{e}_{uv}^{(k)}=\mathrm{MLP}_{\mathrm{enc}}^{(k)}\!\left(\mathbf{a}_{uv}^{(k)}\right)\in\mathbb{R}^{d},
\end{equation}

\subsubsection{Directed edge-aware propagation}
TPN stacks $L$ propagation layers. At layer $l$, for each incoming edge $(u,v)\in\mathcal{E}^{(k)}$, we compute a relation-aware message as
\begin{equation}
\mathbf{m}_{uv}^{(k,l)}
=
\mathrm{MLP}_{\mathrm{msg}}^{(k)}
\!\left(
\left[
\mathbf{h}_u^{(l)} \Vert \mathbf{h}_v^{(l)} \Vert \mathbf{e}_{uv}^{(k)}
\right]
\right),
\end{equation}
where $\Vert$ denotes concatenation. Unlike homogeneous graph encoders that aggregate only neighboring node states, this formulation conditions each message on the source node, the target node, and the embedded edge attributes. As a result, the propagated information is explicitly aware of both timing direction and edge-level physical context.

For node $v$, messages from all incoming neighbors are aggregated by
\begin{equation}
\mathbf{a}_v^{(l)}
=
\sum_{k=0}^{3}\sum_{u\in\mathcal{N}^{(k)}(v)}
\mathbf{m}_{uv}^{(k,l)},
\end{equation}
where
\[
\mathcal{N}^{(k)}(v)=\{u\mid (u,v)\in\mathcal{E}^{(k)}\}
\]
denotes the set of predecessors of $v$ under relation type $k$. We use summation as the aggregation operator because it is permutation-invariant and naturally accumulates the influence of multiple fanin dependencies.

The hidden state of node $v$ is then updated as
\begin{equation}
\mathbf{h}_v^{(l+1)}
=
\mathrm{LayerNorm}
\!\left(
\mathbf{h}_v^{(l)}
+
\mathrm{MLP}_{\mathrm{upd}}
\!\left(
\left[
\mathbf{h}_v^{(l)} \Vert \mathbf{a}_v^{(l)}
\right]
\right)
\right).
\end{equation}
The residual connection improves optimization stability in deep propagation, while layer normalization mitigates scale variation caused by different node degrees and message magnitudes.

This propagation mechanism is consistent with the causal structure of static timing analysis, where the arrival time at node $v$ is determined by its predecessors and the delays on incoming edges:
\begin{equation}
t_v=\max_{u\in\mathrm{pred}(v)}\left(t_u+d_{uv}\right).
\end{equation}
Although TPN does not explicitly hard-code the max operator, its directed edge-aware propagation is designed to serve as a learnable approximation to this timing accumulation process.

\subsubsection{Node-level and graph-level prediction}
After $L$ propagation layers, TPN produces both node-level and graph-level outputs. For each node $v$, a regression head predicts its normalized arrival time:
\begin{equation}
\hat{y}_v=\mathrm{MLP}_{\mathrm{arr}}(\mathbf{h}_v^{(L)}).
\end{equation}
In parallel, a graph-level readout is applied to the final node representations to obtain a global graph embedding
\begin{equation}
\mathbf{h}_{\mathcal{G}}=\mathrm{Readout}\bigl(\{\mathbf{h}_v^{(L)}\mid v\in\mathcal{V}\}\bigr),
\end{equation}
which is then mapped to the predicted critical path delay:
\begin{equation}
\widehat{\mathrm{CPD}}=\mathrm{MLP}_{\mathrm{cpd}}(\mathbf{h}_{\mathcal{G}}).
\end{equation}

The two outputs play complementary roles. The node-level prediction captures the relative timing distribution within a design, which is important for identifying critical nodes and preserving timing order. The graph-level prediction, in contrast, provides supervision on the absolute timing scale of the whole circuit.

\subsubsection{Training objective}
To jointly supervise the two outputs, we define the node-level loss as
\begin{equation}
\mathcal{L}_{\mathrm{node}}
=
\frac{1}{|\mathcal{V}|}
\sum_{v\in\mathcal{V}}
\left(
\hat{y}_v-\frac{t_v}{\mathrm{CPD}}
\right)^2,
\end{equation}
where the target arrival time is normalized by the graph-level CPD to reduce scale variation across circuits. The graph-level loss is defined as
\begin{equation}
\mathcal{L}_{\mathrm{cpd}}
=
\left(
\widehat{\mathrm{CPD}}-\mathrm{CPD}
\right)^2.
\end{equation}
The overall objective is
\begin{equation}
\mathcal{L}
=
\mathcal{L}_{\mathrm{node}}
+
\alpha\,\mathcal{L}_{\mathrm{cpd}},
\end{equation}
where $\alpha>0$ is a scaling coefficient balancing node-level supervision and graph-level supervision. A larger $\alpha$ places more emphasis on recovering the absolute timing scale of the whole graph, whereas a smaller $\alpha$ biases training toward finer modeling of the relative timing distribution within each circuit.


% \section{GAE-enhanced Timing Predictor}

% The focus of our predictor is on estimating the net delay for each source-sink (SS) pair, as illustrated in Fig.~\ref{fig:rrg}. In this context, the source is a virtual node representing the logical output, while the sink is a virtual node corresponding to the logical input.

% The overall flow of our predictor is shown in Fig.~\ref{fig:pf}. To predict the net delay, we extract three key feature sets: (1) topological features from the RRG, (2) distance features derived from the placement of the SS pair, and (3) circuit-level and net structural features from the timing graph. Notably, the features extracted from the RRG are enhanced through a pre-trained GAE. 

% These feature sets are then concatenated into a single feature vector for each SS pair, which is used for both training and inference with an XGBoost-based regressor. The predicted delays are subsequently propagated along the timing graph to calculate the arrival times at each endpoint. 
% The predictor is trained once on multiple designs and directly applied to unseen test circuits without per-design retraining, demonstrating cross-design generalization.
% \begin{figure}[t]
%     \centering
%     \includegraphics[width=\linewidth]{figure/predict_flow.png}
%     \caption{The GAE-enhanced timing predictor.}
%     \label{fig:pf}
% \end{figure}
% \vspace{3pt}
% \subsection{Intra-Tile RRG Construction}

% The full RRG in modern FPGA architectures is extremely large.  
% A complete RRG typically contains over \textbf{150K nodes}, which is beyond the practical input scale of GNN models.
% Fortunately, as shown in Fig.~\ref{fig:structure_rrg}, two structural properties of FPGA routing architectures enable effective graph reduction:

% \begin{itemize}
%     \item \textbf{Redundancy.} After placement, only a small subset of RRG nodes is reachable from mapped sources and sinks. Other nodes are never activated and can be safely removed.

%     \item \textbf{Local modularity and architectural stationarity.}
%     FPGA fabrics consist of repeated tile types (e.g., CLB, DSP, BRAM), each with similar intra-tile routing, forming self-contained subgraphs that are stable across circuits.
% \end{itemize}

% \subsubsection{RRG pruning}
% We perform a forward breadth-first search from valid source nodes and a backward breadth-first search from valid sink nodes. The intersection of the nodes found in both searches defines the valid RRG, while unreachable nodes are removed. On our benchmarks in Section~\ref{exp}, pruning reduced the RRG size by an average factor of \underline{2$\times$}, with a maximum reduction of \underline{4.7$\times$}.

% \subsubsection{Intra-tile RRG extraction}

% In a pruned RRG, we partition the nodes into subgraphs based on their coordinates, creating intra-tile RRGs. Specifically, interconnect nodes are included in all tiles to which they are connected. Each node's features consist of its own attributes, the features of the edges driving it, and the global features of the tile it resides in, as shown in the Table~\ref{tile_feature}. Among the global features, $pins$ represents the number of nodes in the pruned tile, and $bb\_count$ indicates how many net bounding boxes overlap with the tile. In our experiments, each intra-tile RRG has about \underline{100 nodes}, reducing the size by roughly \underline{three orders of magnitude} on average and up to four in the best case.


% \begin{figure}[t]
%     \centering
%     \includegraphics[width=\linewidth]{figure/intra-tile.png}
%     \caption{The structural properties of FPGA RRG.}
%     \label{fig:structure_rrg}
% \end{figure}

% \begin{table}[t]
% \small
% \caption{Feature List of Intra-tile RRG Node}
%     \centering
%     \resizebox{\linewidth}{!}{
% \setlength{\tabcolsep}{2pt}
%     \begin{tabular}{l| c c c}
%         \toprule
%         \textbf{Type} & \textbf{Node} & \textbf{Edge} & \textbf{Global} \\
%         \midrule
%         \textbf{Attributes} & Type, \(x\), \(y\), \(R_\text{switch}\), \(C\) & \(C_\text{in}\), \(C_\text{out}\), \(R_\text{wire}\), Delay & \(pins\), \(bb\_count\) \\
%         \bottomrule
%     \end{tabular}}
%     \label{tile_feature}
% \end{table}


% \subsection{GAE Pre-training}

% We introduce the EdgeFusion Message Passing Neural Networks (MPNNs) to incorporate edge delay into message passing, and use a GAE to enhance node embeddings with broader structural context.

% \subsubsection{EdgeFusion MPNN Propagation Mechanism}
% EdgeFusion MPNN incorporates edge delay information directly into the
% message passing process. Let $h_i$ denote the embedding of node $i$, and
% let $e_{ij}$ denote the delay feature on the RRG edge from node $j$ to
% node $i$. The message and update functions are defined as:
% \begin{equation}
% m_{ij} = \mathrm{MLP}\bigl([\, h_i \,\Vert\, h_j \,\Vert\, e_{ij} \,]\bigr),
% \end{equation}
% \begin{equation}
% h_i^{\text{new}} = h_i \;+\; \sum_{j \in \mathcal{N}(i)} m_{ij}.
% \end{equation}
% Here, $m_{ij}$ is the message sent from neighbor $j$ to node $i$, computed
% jointly from their embeddings and the physical delay feature on the
% connecting routing edge. The updated embedding $h_i^{\text{new}}$
% aggregates messages from all neighbors $\mathcal{N}(i)$, enabling each
% node to integrate both its own state and the delay-dependent structural
% information propagated through adjacent routing edges.

% Since $e_{ij}$ reflects the physical delay along an RRG connection, the message passing process effectively mimics delay propagation over the routing architecture. Because information flows strictly along the RRG topology, the model learns structural properties such as wire connectivity, switch-box patterns, and spatial reachability.

% Congestion is implicitly encoded through the number of incoming messages: nodes with higher fan-in or fan-out accumulate more competing delay signals, enabling the model to associate such patterns with increased timing pressure. Through this formulation, EdgeFusion MPNN jointly captures delay behavior, routing topology, and congestion characteristics in the RRG.

% \subsubsection{GAE Framework}

% GAEs are unsupervised models that learn node representations by encoding a graph into a latent space and reconstructing its structure from the latent embeddings.  
% Let $E_{\theta}$ and $D_{\varphi}$ denote the encoder and decoder, respectively. In our design, both components are implemented based on EdgeFusion MPNN. The encoder maps the input node features into a latent space, and the decoder reconstructs the node features. The entire 

% The reconstruction loss follows the feature-level formulation
% \begin{equation}
% \mathcal{L}_{\text{recon.}}
% = \left\| X - D_{\varphi}\!\left(E_{\theta}(g, X)\right) \right\|_2^2,
% \end{equation}
% which forces the latent space to preserve essential routing structures, enabling the model to learn connectivity patterns, local interactions, and global RRG organization.


% To prevent overly dispersed latent representations, we introduce a Kullback-Leibler (KL) divergence term that regularizes the latent distribution toward a standard normal prior. The complete training objective is
% \begin{equation}
% \mathcal{L}_{\text{total}}
% = \mathcal{L}_{\text{recon.}}
% + \alpha \, \mathcal{L}_{\text{KL}},
% \label{eq:loss}
% \end{equation}
% where $\alpha$ controls the strength of the KL penalty.



% \begin{figure}[t]
%     \centering
%     \includegraphics[width=\linewidth]{figure/gae.png}
%     \caption{GAE Pre-training and Inference.}
%     \label{fig:gnnpd}
% \end{figure}


% \subsection{Feature Engineering}

% For each SS pair, we construct a feature vector that integrates timing-graph connectivity, physical SS-level information, and routing-architecture context. Let 
% $\mathbf{f}^{\text{TG}}$, $\mathbf{f}^{\text{SS}}$, and $\mathbf{f}^{\text{RRG}}$ 
% denote these three groups of features, as illustrated in Fig.~\ref{fig:pf}.

% \subsubsection{Timing-graph features ($\mathbf{f}^{\text{TG}}$).}
% Each SS edge in the timing graph is characterized by local and global connectivity 
% statistics, including node in/out degrees, graph density, and the average/max degree. 
% These indicators correlate with routing complexity, detours, and potential congestion.

% \subsubsection{Physical SS-pair features ($\mathbf{f}^{\text{SS}}$).}
% We extract macroscopic geometric properties including the Manhattan distance between 
% source and sink and the width/height of the enclosing bounding box. 

% \subsubsection{Tile-level and GAE-derived features ($\mathbf{f}^{\text{RRG}}$).}
% As mentioned, each RRG node $v_i$ is mapped to a latent embedding $h_i^{\text{latent}}$.
% For each SS, we extract the embeddings of its source and sink nodes, and
% compute the average embedding of all nodes within the net's bounding box as an approximation along potential routing paths. These three embeddings are then concatenated to form $\mathbf{f}^{\text{RRG}}$.

% These three feature groups capture complementary perspectives: timing-graph connectivity models logical complexity, SS-level features reflect geometric distance, and GAE embeddings encode routing-architecture constraints. Their concatenation therefore preserves multi-scale information essential for net delay prediction.

% \subsection{XGBoost-based AT Predictor}
% We construct a fused feature vector $\mathbf{f}_{\text{SS}} = 
% [\mathbf{f}^{\text{TG}} \Vert \mathbf{f}^{\text{SS}} \Vert \mathbf{f}^{\text{RRG}}]$ for each SS pair, and use XGBoost~\cite{XGBoost} to predict the post-routing net delay, with labels from VTR~\cite{vtr9}.



% After predicting net delays for all SS edges, the arrival time $AT(v_t)$ of each timing node $v_t$ is then computed by a single forward traversal using the standard STA recurrence
% \begin{equation}
% AT(v_t) = \max_{u_t \in \mathit{pred}(v_t)} \bigl( AT(u_t) + \text{delay}(u_t \rightarrow v_t) \bigr),
% \end{equation}
% where $\mathit{pred}(v_t)$ denotes the predecessors of $v_t$. Nodes with the largest arrival time define the critical endpoints, and the critical path is recovered by back-tracing predecessor links. 




% \begin{figure}[t]
%     \centering
%     \begin{subfigure}{0.51\linewidth}
%         \centering
%         \includegraphics[width=\linewidth]{figure/br.png}
%         \caption{Backward retiming.}
%         \label{fig:retiming_backward}
%     \end{subfigure}
%     %\vspace{0.5pt}
%     \begin{subfigure}{0.48\linewidth}
%         \centering
%         \includegraphics[width=\linewidth]{figure/fr.png}
%         \caption{Forward retiming.}
%         \label{fig:retiming_forward}
%     \end{subfigure}

%     \caption{Two fundamental retiming moves in ABC.}
%     \label{fig:retiming_moves}
% \end{figure}




\section{Experiments}
\label{exp}

\subsection{Experimental Setup}

Our dataset consists of 45 circuits drawn from the VTR~\cite{vtr9} and MCNC~\cite{mcnc} benchmark suites, targeting the academic VTR FPGA architecture \texttt{k6\_frac\_N10\_frac\_chain\_mem32K\_40nm}, which is derived from the Stratix IV family. Among these designs, 26 are used for training, while the remaining 19 serve as unseen test circuits. The training set provides approximately 30k intra-tile RRG graph examples for GAE pre-training and about 1M SS pair samples for training the XGBoost-based delay regressor. For each SS pair, the target label is the post-routing net delay extracted from VTR’s static timing analysis engine.

The predictor is optimized using a mean-squared error (MSE) objective. Within each training design, we further apply a 90/10 sample-level split to form validation subsets used for hyperparameter tuning and early stopping. 


The complete processing and prediction flow is summarized in Fig.~\ref{fig:framework}, and is implemented on VTR~\cite{vtr9}, Parmys~\cite{vtr9}, and ABC~\cite{abc}. Data extraction is now integrated within the modified VTR9 framework. The extracted data is then transferred to Python via the \texttt{cnpy} library for further processing. Graph data handling and prediction tasks are carried out using the DGL~\cite{dgl} and XGBoost libraries~\cite{XGBoost}.

All experiments are conducted on a Linux server with an Intel Xeon Platinum 8354H CPU (single-core execution) and no GPU acceleration. All circuits before and after retiming are verified for functional equivalence using ABC’s \texttt{dsec} command~\cite{ref:abc_dsec}.


\subsection{Timing Predictor Evaluation}
\subsubsection{GAE Pretraining}



We compare our proposed EdgeFusion MPNN with three widely used models in the graph neural network domain: GAT~\cite{gat}, GCNII~\cite{gcn2}, and GINE~\cite{gine}. All models are trained on the same pre-training dataset, and we conduct extensive hyperparameter tuning, exploring factors such as the network depth, the learning rate, and the hidden layer dimensions. The reconstruction loss is computed according to Equation~\ref{eq:loss}, with a regularization parameter $\alpha = 0.2$. The results are presented in Table~\ref{tab:gae}.

As demonstrated in the results, EdgeFusion MPNN exhibits a significantly lower reconstruction loss compared to other models, highlighting its superior ability to capture the underlying graph structure and preserve node-level features with greater accuracy. As detailed in Table~\ref{tile_feature}, edge features are incorporated at the source node, which indicates that the model also effectively retains edge delay information.
Moreover, EdgeFusion MPNN also exhibits a notably low KL loss, suggesting that the learned latent representations are well-regularized. This indicates that the model prevents over-dispersion of node embeddings, which is crucial for prediction.


The final hyperparameters are: hidden dimension = 64, latent dimension = 8, number of layers = 2, dropout rate = 0.2.


\subsubsection{Ablation Study}

We designed four comparative experiments for the AT prediction: 1) w/o $f\_{ss}$ represents the traditional prediction that does not consider net delay (sets them to 0). 2) w/o $\mathbf{f}^{\text{GAE}}$, w/o $\mathbf{f}^{\text{SS}}$, and w/o $\mathbf{f}^{\text{TG}}$ represent the results after removing specific features. Since our goal is to identify critical paths, we focus only on the arrival time at the endpoints of each path. Each experiment is individually tuned using the validation set. 
Our final predictions are made with key hyperparameters: max\_depth=7, eta=0.05, subsample=0.7, colsample\_bytree=0.7, min\_child\_weight=10.

We evaluate the results using the coefficient of determination ($R^2$), where higher values indicate better performance. A value of 1 indicates perfect prediction, while values less than 0 suggest that predicting the average value would yield better results. The final results are presented in Table~\ref{tab:cp_predict}.

Our model achieves a prediction accuracy of 0.90, whereas traditional predictions that do not consider net delay often fail to accurately identify the true endpoints. Through ablation experiments, we demonstrate the necessity of each feature. For instance, removing key features such as $f_{ss}$ and $\mathbf{f}^{\text{GAE}}$ significantly reduces prediction accuracy, further validating the contribution of these features to improving performance.
Notably, the \textit{blob\_merge} circuit has few path endpoints with similar delay values, making it more challenging for prediction. Despite this, our predictor still performs well, demonstrating its robustness even in such difficult cases.

% As shown in Table~\ref{tab:cp_predict}, our model achieves a prediction accuracy of 0.92, compared to 0.19 without net delay. This accuracy is particularly high for large-scale circuits with longer paths, such as \texttt{mcml} and \texttt{sha}. For circuits where most paths share similar logic levels and net delays, such as \texttt{s38584.1} and \texttt{mkPktMerge}, the model's performance is limited. Additionally, while most circuits allow for accurate identification of the top 10 longest paths, circuits involved in packet processing, like the \texttt{mk} series, face challenges due to multiple paths having similar lengths, making it difficult for the model to distinguish between them.

\begin{table}[!t]
\centering
\caption{The Loss of GAE on the Test Dataset.}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{c |c c  c c  }  
\toprule
Layer Type& EdgeFusion MPNN& GAT~\cite{gat}& GINE~\cite{gine}& GCNII~\cite{gcn2}\\
 \midrule
Val Loss& 1.43& 17.48& 20.09& 67.69\\ \bottomrule
\end{tabular}
\label{tab:gae}
\vspace{10pt}
\end{table}

\begin{table}[t]
\centering
\caption{The Accuracy of AT Prediction on the Test Dataset.}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{c |c c  c c  c }  
\toprule
Design& w/o $\mathbf{f}_{\text{SS}}$& w/o $\mathbf{f}^{\text{GAE}}$& w/o $\mathbf{f}^{\text{SS}}$& w/o $\mathbf{f}^{\text{TG}}$ & Ours\\
 \midrule
bgm
& 0.64
& 0.85
& 0.83
& 0.83
& 0.97
\\
blob\_merge
& -2.67
& 0.25
& 0.20 
& 0.13
& 0.64\\
boundtop
& -1.5
& 0.77
& 0.79
& 0.76
& 0.83
\\
ch\_intrinsics
& -1.19
& 0.81
& 0.81
& 0.81
& 0.85
\\
diffeq1
& 0.73
& 0.96
& 0.97
& 0.98
& 0.98
\\
diffeq2
& 0.80 
& 0.98
& 0.99
& 0.98
& 0.99
\\
dsip
& -1.39
& 0.79
& 0.65
& 0.56
& 0.86
\\
frisc
& 0.11
& 0.96
& 0.94
& 0.75
& 0.96
\\
mcml
& 0.78
& 0.85
& 0.88
& 0.89
& 0.96
\\
mk1$^*$ 
& -4.61
& 0.49
& 0.73
& 0.43
& 0.84
\\
mk2$^*$ 
& -2.71
& 0.80 
& 0.81
& 0.82
& 0.87
\\
mk3$^*$ 
& 0.02
& 0.87
& 0.87
& 0.86
& 0.88
\\
or1200
& -1.36
& 0.97
& 0.96
& 0.88
& 0.97
\\
raygentop
& -0.29
& 0.88
& 0.86
& 0.83
& 0.88
\\
s38417
& 0.09
& 0.92
& 0.92
& 0.89
& 0.94
\\
s38584.1
& -1.07 
& 0.64
& 0.69
& 0.67
& 0.82
\\
sha
& -0.85
& 0.97
& 0.93
& 0.91
& 0.98
\\
spree
& 0.87 
& 0.99
& 0.99
& 0.98
& 0.99
\\
tseng
& 0.33
& 0.90 
& 0.90 
& 0.82
& 0.92
\\ \midrule
Avg. Test& -0.70& 0.82& 0.83& 0.78& 0.90\\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item$^*$ mk1: mkDelayWorker32B. mk2: mkPktMerge. mk3: mkSMAdapter4B.
\end{tablenotes}
\label{tab:cp_predict}
\vspace{3pt}
\end{table}



\begin{table*}[t]
\centering
\renewcommand{\arraystretch}{0.9}
\caption{Performance Comparison of Our Method against Baseline and ABC Retiming on the Test Dataset.}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{c|ccc|cccc|cccc}
\toprule
\multirow{2}{*}{\textbf{Benchmark}} & \multicolumn{3}{c|}{\textbf{Baseline}}                                       & \multicolumn{4}{c|}{\textbf{ABC Retiming}}                                                          & \multicolumn{4}{c}{\textbf{Our method}}                                                            \\
                           & \#Node               & \#Level              & $F_\text{max}$(MHz)                 & \#Node               & \#Level              & $F_\text{max}$(MHz)                  & Impr.& \#Node               & \#Level              & $F_\text{max}$(MHz)                  & Impr.\\
                           \midrule
bgm                        & 76213                & 95                   & 58.41              & -& -& -& -& 76673                & 87                   & 60.07 
& 2.85\%\\
blob\_merge                & 40081                & 102                  & 93.29              & 40081& 102& 96.46              & 3.41\%& 40089                & 95                   & 104.34 
& 11.84\%\\
boundtop                   & 376                  & 9                    & 456.73              & 376                  & 9                    & 446.63              & -2.21\%& 378                  & 9                    & 455.15 
& -0.34\%\\
ch\_intrinsics             & 346                  & 15                   & 412.43              & 346                  & 15                   & 452.85               & 9.80\%& 346                  & 12                   & 437.29 
& 6.03\%\\
diffeq1                    & 2749                 & 77                   & 40.20              & 2749                 & 75                   & 124.19              & 208.96\%& 2749                 & 75                   & 124.19 
& 208.96\%\\
diffeq2                    & 2196                 & 75                   & 54.73              & 2203                 & 65                   & 135.45              & 147.47\%& 2198                 & 69                   & 116.69 
& 113.19\%\\
mcml                       & 407351               & 397                  & 14.46              & -                & -& -& -&                      407397&                      397
& 20.09 
& 38.88\%\\
mkDelayWorker32B           & 1362                 & 21                   & 125.32              & 1362                 & 21                   & 76.41              & -39.03\%& 1362& 21                   & 76.41 
& -39.03\%\\
mkPktMerge                 & 550                  & 5                    & 226.65             & 550                  & 5                    & 313.09               & 38.14\%& 550                  & 5                    & 285.05 
& 25.76\%\\
mkSMAdapter4B              & 4890                 & 35                   & 166.94              & 4885                 & 35                   & 211.73              & 26.84\%& 4898                 & 35                   & 223.12 
& 33.65\%\\
or1200                     & 12833                & 148                  & 68.46              & 12878                & 143                  & 76.98              & 12.44\%& 12845                & 148                  & 73.92 
& 7.97\%\\
raygentop                  & 4321                 & 41                   & 165.49              & -                & -& -& -& 4325                 & 30                   & 254.79 
& 53.96\%\\
sha                        & 11793                & 205                  & 70.21              & 11835                & 98                   & 75.77              & 7.90\%& 11815                & 94                   & 76.73 
& 9.28\%\\
spree                      & 3787                 & 108                  & 25.05              & 3787                 & 108                  & 25.36              & 1.23\%& 3789                 & 109                  & 24.97 
& -0.33\%\\
frisc                      &                      8011&                      66
& 75.08              &                      8011&                      66
& 72.63              & -3.26\%&                      8015&                      61
& 85.92 
& 14.44\%\\
dsip                       &                      3176&                      10
& 345.68              &                      3176&                      10
& 362.09              & 4.75\%&                      3176&                      8
& 424.67 
& 22.85\%\\
s38417                     &                      13870&                      28
& 121.34              &                      13885&                      37
& 107.42              & -11.47\%&                      13885&                      29
& 130.64 
& 7.67\%\\
s38584.1                   &                      12196&                      24
& 141.47              &                      12196&                      24
& 129.38              & -8.55\%&                      12204&                      22
& 150.36 
& 6.28\%\\
tseng                      &                      2501&                      45& 147.46              &                      2501&                      45& 139.82              & -5.18\%&                      2506&                      39& 156.76 
& 6.30\%\\     
\bottomrule
\end{tabular}
\label{tab:retime}
\begin{tablenotes}
The average $F_{max}$ improvement for ABC Retiming is 19.40\%, while our method achieves 27.24\%.
\end{tablenotes}
\vspace{2pt}
\end{table*}



\subsubsection{Runtime Speedup}

We measure the runtime required for pre-routing delay prediction and its speedup over invoking VTR routing and STA. Fig.~\ref{fig:speedup} summarizes the results, where warmer colors indicate designs with longer VTR routing time and cooler colors correspond to smaller, faster designs. As depicted, our method achieves an average speedup of $\mathbf{10.18\times}$, with a maximum of $\mathbf{56.5\times}$ on large designs such as \textit{mcml}. Even on small designs, the predictor still achieves 2–3$\times$ acceleration.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figure/speedup.png}
    \caption{Prediction Speedup Relative to VTR flow.}
    \label{fig:speedup}
    \vspace{3pt}
\end{figure}


\section{Application: Predictor-Guided Retiming}
To achieve physical-aware timing optimization, our framework enhances the heuristic retiming algorithm in ABC~\cite{abc} by addressing its primary limitation: the reliance on a physically-unaware, \textbf{unit-delay model}. In this model, each logic gate is assumed to have an identical delay, allowing the engine to efficiently identify a logical critical path. ABC then iteratively applies forward and backward retiming moves to optimize this path, which involve shifting registers across gates as shown in Fig.~\ref{fig:retiming_moves}.



While computationally efficient, this purely logical approach is inherently flawed. The critical path identified by ABC often fails to correspond to the true performance-limiting path after physical place-and-route (P\&R), as it neglects crucial physical effects like wire delays and placement congestion. This discrepancy, known as poor timing correlation, leads to misdirected optimization efforts and suboptimal post-P\&R timing results. Our work directly targets this gap by replacing the simplistic unit-delay assumption with accurate, physically-informed timing predictions.



\makeatletter
\newcommand{\COMMENT}[1]{\STATE \textcolor{blue}{// #1}}
\makeatother
\begin{algorithm}[t]
\caption{Timing Predictor-enhanced Retiming within ABC}
\label{alg:gnn_retime_final}
\small
\begin{algorithmic}[1]
    \Statex \textbf{Input:} Initial netlist $N_{init}$
    \Statex \textbf{Output:} Optimized netlist $N_{opt}$
    \State $N_{opt} \gets N_{init}$
    \Loop\\
\COMMENT{1.Predict critical path via GNN.}
        \State $P_{crit} \gets \text{PredictCriticalPath}(N_{opt})$\\
\COMMENT{2. Find the best retiming move on the path.}
        \State $best\_move \gets \text{null}$
        \State $max\_gain \gets 0$
        \For{\textbf{each} node $v \in P_{crit}$}
            \For{\textbf{each} move $m \in \{\text{FWD, BWD}\}$}
                \If{$\text{IsFeasible}(N_{opt}, v, m)$}
                    \State $gain \gets \text{CalculateLocalGain}(N_{opt}, v, m)$
                    \If{$gain > max\_gain$}
                        \State $max\_gain \gets gain$
                        \State $best\_move \gets (v, m)$
                    \EndIf
                \EndIf
            \EndFor
        \EndFor\\
\COMMENT{3. Convergence occurs when no profitable move is found.}
        \If{$best\_move = \text{null}$}
            \State \textbf{break}
        \Else
            \State $N_{opt} \gets \text{ApplyMove}(N_{opt}, best\_move)$
        \EndIf
    \EndLoop
\end{algorithmic}
\end{algorithm}

The core of our framework is the replacement of the inaccurate unit-delay model with a highly accurate, physical-aware timing predictor. As detailed in Algorithm~\ref{alg:gnn_retime_final}, our approach integrates this predictive model into an iterative optimization loop with three key stages per iteration: Predict, Evaluate, and Apply.
First, in the Predict stage, we invoke our timing predictor to identify the top-$k$ post-routing critical paths, $P_{crit}$ (line 4). This is the key to our contribution: by substituting the simplistic unit-delay abstraction with physically-aware data, we ensure that optimization is always focused on the true performance bottlenecks of the design.
Next, in the Evaluate stage, the algorithm leverages ABC’s proven heuristic to find the most profitable retiming move. It systematically considers feasible forward and backward moves for nodes exclusively along the predicted critical paths ($P_{crit}$), calculating a gain for each potential transformation (lines 5–18). This strategy preserves the efficiency of the heuristic search by constraining it to the most impactful regions of the circuit.
Finally, the single move yielding the maximum gain is selected and applied to the netlist (line 23). This iterative process continues until no further timing improvements are possible (lines 19–20), at which point the converged, optimized netlist $N_{opt}$ is returned.


\subsection{Physical-aware Retiming}





Table~\ref{tab:retime} compares the baseline flow, the standard ABC retiming, and our proposed method. In this table, \#Node and \#Level represent the number of AIG nodes and the logic depth reported by ABC, which describe the circuit size and the depth of the longest combinational path, respectively.
The results show that our method achieves a 27\% improvement in maximum frequency (\(F_\text{max}\)) over the baseline, outperforming the 19\% improvement from the standard ABC retiming flow. For benchmarks marked with "-", the ABC retime command failed due to duplicate CI and CO errors, yielding no results. This performance gain is due to our method's use of physical information for more realistic retiming decisions, and by bypassing the time-consuming routing step of traditional optimization flows. 
%It is also worth noting that although our method achieves a higher average improvement in \(F_\text{max}\) compared to ABC retiming, the resulting netlists do not always exhibit fewer logic levels, highlighting the inherent discrepancy between logic-level optimization objectives and physical timing behavior on FPGAs.

The benefits of our physical-aware strategy are particularly pronounced for circuits where long, complex routing paths are expected to be the primary performance limiters on an FPGA ( \textit{dsip}, \textit{raygentop}, \textit{blob\_merge}). 
%Furthermore, for several designs where ABC retiming results in a performance degradation ( \texttt{frisc}, \texttt{s38417}), our method still manages to deliver a significant performance uplift. 
However, for some circuits dominated by dense logic or a high concentration of macro-blocks, such as \textit{diffeq2}, their critical paths are inherently logic-bound rather than routing-limited. In such cases, our optimization strategy, which primarily targets physical path delays, is naturally less effective than a purely logical approach that directly addresses logic depth. Despite this, the overall results confirm that incorporating physical information early in the flow provides a substantial net benefit, leading to higher-quality final designs.

Table~\ref{tab:retiming_comparison} compares our approach with prior works. Unlike the work~\cite{ref_iterret}, which does not integrate timing prediction and instead relies on the entire flow to obtain timing information via STA, our method integrates timing prediction directly into the retiming process. This integration allows our approach to achieve better results on a broader set of benchmarks.
Additionally, our method uses predicted timing information at the post-placement stage, providing more accurate and efficient timing optimization compared to the rough estimations~\cite{ref:rta} or simple timing model~\cite{b3}.

\begin{table}[!t]
\centering
\caption{Comparison with Prior Works}
\resizebox{\linewidth}{!}{
\setlength{\tabcolsep}{1pt}
\label{tab:retiming_comparison}
\begin{tabular}{c|ccccc}
\toprule
                      & \textbf{FPGA/ASIC}  & \begin{tabular}[c]{@{}c@{}}\textbf{Iterative}\\ \textbf{Support}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{Timing}\\ \textbf{Predict}\end{tabular} & \begin{tabular}[c]{@{}c@{}}\textbf{Retiming}\\ \textbf{Stage}\end{tabular} & \textbf{Delay Model Used}       \\ \midrule
ABC~\cite{ref:abcret} & FPGA, ASIC & \XSolidBrush                                                & \XSolidBrush                                             & Before Mapping                                             & Logic Depth          \\
~\cite{ref_iterret}   & FPGA       & \CheckmarkBold                                              & \XSolidBrush                                             & Post STA                                                 & STA                    \\
~\cite{ref:latret}    & ASIC       & \XSolidBrush                                                & \XSolidBrush                                             & Gate-Level                                               & STA                    \\
\cite{b3}               & FPGA       & \XSolidBrush                                                & \XSolidBrush                                             & Post Routing                                             & Simple Timing Model    \\
RTA~\cite{ref:rta}    & ASIC       & \XSolidBrush                                                & \XSolidBrush                                             & Post Placement                                           & Rough Estimation       \\
Ours                  & FPGA       & \CheckmarkBold                                              & \CheckmarkBold                                           & Post Placement                                                 & Predicted Timing Info. \\ \bottomrule
\end{tabular}}
\vspace{3pt}
\end{table}

\section{Conclusion}

We proposed a physical-aware retiming method that leverages a timing predictor to accelerate the process. By effectively extracting features such as congestion, topology, and geometric distance, the predictor enables accurate pre-routing predictions, which enhance the efficiency of retiming. Our approach demonstrates significant performance improvements over traditional methods.




\bibliographystyle{IEEEtran}
\bibliography{DAC-GNNRet}


%%
%% If your work has an appendix, this is the place to put it.
\appendix



\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex'.
