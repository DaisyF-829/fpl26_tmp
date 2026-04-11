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

\usepackage{algpseudocode}
\usepackage{textcomp}
\usepackage{bbding}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}


\usepackage{array}
\newcolumntype{V}{!{\hspace{1.8pt}\vrule width 0.45pt\hspace{1.8pt}}}
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{1.08}

\algrenewcommand\alglinenumber[1]{\scriptsize #1:}

\pagestyle{empty}
\setlength{\textfloatsep}{3pt plus 1pt minus 1pt}
\setlength{\floatsep}{6pt plus 1pt minus 1pt}
\setlength{\intextsep}{6pt plus 1pt minus 1pt}

\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}


\begin{document}

\title{PASTE: A \underline{P}hysical-\underline{A}ware \underline{S}urrogate for FPGA Pre-Routing \underline{T}iming \underline{E}stimation}
\maketitle

\begin{abstract}



Accurate pre-routing timing estimation is a fundamental challenge in FPGA design, since the final path delay is jointly determined by the timing graph, the placement, and the architecture-constrained routing fabric. Existing methods target only coarse-grained proxies, rather than fine-grained node-level arrival-time prediction.
In this paper, we present PASTE, a physical-aware surrogate for FPGA pre-routing timing estimation. PASTE constructs physically informative features on the post-placement timing graph, including the placement geometry, the post-placement timing prior, and routing-density signals derived from a design-aware pruned routing resource graph. Building on these features, we propose the Timing Propagation Network (TPN), a two-stage graph neural network that jointly predicts node-level arrival times and critical path delay. The first stage uses heterogeneous message passing to learn local multi-type timing dependencies, while the second stage performs edge-gated max propagation to model the long-range max-plus accumulation underlying static timing analysis. 
Both stages operate directly on the timing graph without explicit per-net routing-aware modeling, improving scalability to large designs while enabling sub-second inference.
Implemented in VTR, PASTE achieves a node-level MAPE of 8.48\% and a graph-level CPD MAPE of 8.28\% under the circuit-level setting on the test set.
As a downstream validation, integrating PASTE into a predictor-guided retiming flow achieves an average $F_{\max}$ improvement of 30.83\% over the unoptimized baseline, outperforming logic-only ABC retiming by 4.26\%. These results demonstrate that PASTE serves as an effective timing surrogate for early-stage timing analysis and timing-driven FPGA optimization.

\end{abstract}


\begin{IEEEkeywords}
GNN, STA, Timing Prediction
\end{IEEEkeywords}







\section{Introduction}

Field-programmable gate arrays (FPGAs) have become an important platform
for hardware acceleration due to their flexibility and strong
domain-specific performance. As FPGA designs continue to grow in scale
and complexity, timing closure has become a major challenge in the CAD
flow. In practice, many timing-driven optimizations must be performed
before detailed routing, including logic restructuring, placement
refinement, and sequential optimization. Therefore, accurate
pre-routing timing prediction is highly desirable, since it can provide
early timing guidance without incurring the prohibitive cost of
iterative routing.

Pre-routing timing prediction in FPGAs is particularly challenging
because timing is jointly determined by logic delay, placement, and the
architecture-constrained routing fabric. In particular, routing delay
can contribute a substantial portion of critical-path delay in
modern FPGAs~\cite{ref:route_delay}, making logic-only estimation
insufficient to identify true timing bottlenecks. As summarized
in Table~\ref{tab:related}, most of the prior FPGA learning-based methods focus
on indirect objectives or restricted prediction targets, rather than
directly predicting arrival time for timing analysis. A recent work by
Dai \emph{et al.}~\cite{dai2025rrg_gae} took an important step toward
FPGA pre-routing arrival-time prediction. However, to incorporate
routing-resource information, that framework performs explicit
per-net routing-aware modeling over all nets in the design, leading to
overhead that scales linearly with the number of nets. In addition, its routing-aware representation
learning and timing prediction are not optimized jointly under a single
end-to-end arrival-time objective.

\begin{table}[t]
\centering
\caption{Related works on pre-routing timing prediction and related learning tasks.}
\label{tab:related}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{c|cccc}
\toprule
\textbf{Reference} &
\textbf{Domain} &
\textbf{Granularity} &
\begin{tabular}[c]{@{}c@{}}\textbf{Circuit-}\\\textbf{Aware}\end{tabular} &
\textbf{Prediction Target} \\
\midrule
\cite{WCPNet,wirelength} & FPGA & Circuit-level & \CheckmarkBold & Wirelength \\
\cite{bbcong,conghls,fGREP} & FPGA & Region-level & \CheckmarkBold & Congestion \\
\cite{dl_routability,sta_rout_mux} & FPGA & Circuit-level & \CheckmarkBold & Routability \\
\cite{hlspd} & FPGA & Op-level & \CheckmarkBold & Operation delay \\
\cite{fpga_ml_delay} & FPGA & Net-level & \XSolidBrush & Net delay \\
\cite{air} & FPGA & Net-level & \XSolidBrush & Routing cost \\
\cite{timinggcn,letime,gattiming,preroutegnn} & ASIC & Cell-level & \CheckmarkBold & \textbf{Arrival time} \\
\cite{dai2025rrg_gae} & FPGA & Node-level & \CheckmarkBold & \textbf{Arrival time} \\
Ours & FPGA & Node-level & \CheckmarkBold & \textbf{Arrival time} \\
\bottomrule
\end{tabular}%
}
\end{table}

In this work, we propose PASTE, a physical-aware surrogate for
FPGA pre-routing arrival-time prediction that operates on the
post-placement timing graph. By augmenting the timing graph with
routing-density signals derived from a design-aware pruned RRG
and modeling timing propagation through a two-stage GNN, PASTE
achieves accurate node-level prediction without explicit per-net
routing-aware modeling. The main contributions are summarized as
follows: 
\begin{itemize}
\item We construct physically informative features on the
post-placement timing graph, including a normalized timing
prior and routing-density signals derived from a design-aware
pruned RRG, to capture routing-induced timing variation
beyond pure logic topology.

\item We propose the Timing Propagation Network (TPN), a
two-stage GNN that combines heterogeneous message passing
with edge-gated max propagation, explicitly mirroring
the max-plus semantics of static timing analysis. TPN
achieves sub-second
inference on designs with over 400k timing nodes.

\item Experiments on 48 benchmarks show that PASTE achieves
a node-level MAPE of 8.48\% under the circuit-level setting.
As a downstream application, PASTE-guided retiming improves
$F_{\max}$ by 30.83\% over the unoptimized baseline,
outperforming ABC retiming.
\end{itemize} 

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figure/framework.png}
    \caption{Overview of the proposed PASTE framework, including physical-aware feature construction and timing prediction on the post-placement timing graph.}
    \label{fig:framework}
    \vspace{4pt}
\end{figure}



\section{Background}



\subsection{Pre-Routing Timing Estimation in FPGA Design}

Existing FPGA pre-routing timing estimation methods can be divided into three categories. The first category predicts indirect routing-related proxies from placement-derived features. Representative examples include wirelength prediction~\cite{wirelength}, congestion estimation~\cite{bbcong,conghls}, routability prediction~\cite{dl_routability}, and MUX-usage modeling~\cite{fGREP,sta_rout_mux}. Although useful for global quality assessment, these methods do not directly provide node-level timing information for critical-path analysis.

The second category predicts delay for specific targets. Hu \emph{et al.}~\cite{lutdelay} use machine learning to estimate delay during FPGA technology mapping. Ustun \emph{et al.}~\cite{hlspd} use graph neural networks to predict operation-level delay in FPGA HLS. Although effective in their target scenarios, these methods address different abstraction levels and are not directly applicable to generate node-level timing estimation on placed netlists.

The third category uses heuristic routing-cost estimation in FPGA CAD tools. For example, VTR uses architecture-aware routing lookahead to estimate delay and congestion during maze routing~\cite{air,vtr9}. These estimates are designed to guide routing search rather than to serve as accurate timing surrogates for node-level arrival-time prediction or critical-path analysis. 

Overall, existing FPGA methods either predict coarse proxies, address restricted delay subproblems, or provide routing heuristics. They do not provide general node-level timing prediction on placed netlists, which limits their ability to identify critical paths before detailed routing.

\subsection{GNN-Based Timing Prediction}

Graph neural networks (GNNs) learn node representations by aggregating
information from graph neighbors~\cite{gnn}. A general GNN layer can be
written as
\begin{equation}
\mathbf{h}_v^{(l+1)}
=
\phi\!\left(
\mathbf{h}_v^{(l)},
\operatorname{AGG}_{u\in\mathcal{N}(v)}
\psi\!\left(\mathbf{h}_v^{(l)},\mathbf{h}_u^{(l)},\mathbf{e}_{uv}\right)
\right),
\end{equation}
where \(\mathbf{h}_v^{(l)}\) is the representation of node \(v\) at
layer \(l\), \(\mathcal{N}(v)\) denotes its neighbors, and
\(\mathbf{e}_{uv}\) denotes edge features.

This formulation is suitable for timing analysis because arrival time is
propagated along directed timing arcs and depends on both local delays
and graph topology. GNN-based timing predictors have been studied in
ASIC design~\cite{timinggcn,letime,gattiming,preroutegnn}. However, FPGA
timing prediction differs fundamentally from ASIC timing prediction. In
ASIC flows, a net is often approximated as a geometric interconnect
whose delay mainly depends on placement. In
contrast, an FPGA net is routed through a fixed routing fabric by traversing
programmable switches, as illustrated in
Fig.~\ref{fig:rrg}. The routing architecture is typically modeled as a
routing resource graph (RRG), where nodes correspond to routing
resources or pins and edges denote programmable connections. As a result, 
the net delay is strongly influenced by switch-box topology and
local congestion, rather than by geometric distance alone. Based on the RRG, a timing graph can be further constructed for timing analysis, as illustrated in Fig.~\ref{fig:rrg}(c). As can be observed from the timing graph, the total delay of a complete timing path typically consists of three components: the clock-to-Q delay, the net delay, and the combinational logic delay.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{figure/rrg.png}
    \caption{Mapping physical routing elements to RRG and the timing graph.}
    \label{fig:rrg}
    \vspace{4pt}
\end{figure}


\section{Physical-Aware Timing Predictor}
As illustrated in Fig.~\ref{fig:framework}, PASTE takes the placement, RRG, and timing graph of a placed design as input. Placement and RRG information are first used to construct physically informative node and edge features, which are then associated with the timing graph. The resulting physical-aware timing graph is finally processed by the proposed Timing Propagation Network (TPN) to predict node-level arrival times and graph-level CPD.


\subsection{RRG Pruning}
\label{sec:rrg_pruning}

The original RRG contains all routing resources and programmable
connections defined by the FPGA architecture. Features
computed on the RRG can reflect architecture-dependent differences in
interconnect resources and routing topology.
However, if routing-density features are computed on the
full RRG, many resources irrelevant to the current placed design may
also be counted, which can distort the resulting congestion-related
statistics. For example, a design may heavily occupy CLB-related regions
while barely using memory blocks. If memory-related routing resources
are still included in the density statistics, they can enlarge the
effective resource space and artificially dilute the estimated
congestion. We therefore prune the RRG before feature extraction.

Algorithm~\ref{alg:rrg_pruning} summarizes the proposed RRG pruning
procedure. We first map each timing node to its corresponding RRG source
or sink node. Starting from these valid source and sink nodes, we retain
only those RRG nodes that are reachable from valid sources and can
further reach valid sinks. In this way, the pruned RRG preserves only
the routing resources that are potentially relevant to the current
placed design. The resulting valid node set and induced valid edge set
are denoted by \(V_{\mathrm{rr}}^{\mathrm{valid}}\) and
\(E_{\mathrm{rr}}^{\mathrm{valid}}\), respectively. All routing-density
features in the subsequent sections are computed on this pruned RRG.

\begin{algorithm}[t]
\caption{RRG Pruning via Breadth-First Search}
\label{alg:rrg_pruning}
\small
\begin{algorithmic}[1]
\Require Timing-node set $V_t$, RRG node set $V_{\mathrm{rr}}$, RRG edge set $E_{\mathrm{rr}}$
\Ensure Valid RRG node set $V_{\mathrm{rr}}^{\mathrm{valid}}$, valid RRG edge set $E_{\mathrm{rr}}^{\mathrm{valid}}$

\State $(V_{\mathrm{rr,src}}^{\mathrm{valid}},\,V_{\mathrm{rr,sink}}^{\mathrm{valid}})\gets \mathrm{MAP}(V_t)$
\State $S_{\mathrm{valid}} \gets V_{\mathrm{rr,src}}^{\mathrm{valid}}$
\State $T_{\mathrm{valid}} \gets V_{\mathrm{rr,sink}}^{\mathrm{valid}}$
\State $S_{\mathrm{valid}} \gets S_{\mathrm{valid}} \cup \mathrm{BFS\text{-}F}(S_{\mathrm{valid}}, V_{\mathrm{rr}}, E_{\mathrm{rr}})$
\State $T_{\mathrm{valid}} \gets T_{\mathrm{valid}} \cup \mathrm{BFS\text{-}B}(T_{\mathrm{valid}}, V_{\mathrm{rr}}, E_{\mathrm{rr}})$
\State $V_{\mathrm{rr}}^{\mathrm{valid}} \gets S_{\mathrm{valid}} \cap T_{\mathrm{valid}}$
\State $E_{\mathrm{rr}}^{\mathrm{valid}} \gets \{(v_{\mathrm{src}},v_{\mathrm{dst}})\in E_{\mathrm{rr}} \mid v_{\mathrm{src}},v_{\mathrm{dst}}\in V_{\mathrm{rr}}^{\mathrm{valid}}\}$\\
\Return $V_{\mathrm{rr}}^{\mathrm{valid}}, E_{\mathrm{rr}}^{\mathrm{valid}}$
\end{algorithmic}
\end{algorithm}

\subsection{Timing Graph Construction}

Following the timing-graph abstraction used in VTR~\cite{vtr9}, we
model each design as a directed timing graph
\(
\mathcal{G}_{\mathrm{tg}}=(\mathcal{V}_{\mathrm{tg}},\mathcal{E}_{\mathrm{tg}}),
\)
where each timing node is represented as a graph node and each timing
edge is represented as a directed edge describing a valid timing
dependency between two timing nodes. In VTR, timing nodes and timing edges 
correspond to \texttt{tnode} and \texttt{tedge}, respectively, and their 
types used in this work are summarized in Table~\ref{tab:tg_types}.

This
timing graph is maintained and updated throughout the CAD flow as timing
information becomes more accurate. In particular, after placement, VTR
provides a timing graph whose edge delays already incorporate
logic-delay information and whose nodes carry placement-stage timing
annotations. These post-placement timing quantities provide useful prior
knowledge before detailed routing. In this work, we perform pre-routing
timing prediction on this post-placement timing graph.

\begin{table}[t]
\centering
\caption{Timing node and edge types in the timing graph.}
\label{tab:tg_types}
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{p{0.48\linewidth} p{0.38\linewidth}}
\toprule
\textbf{Type} & \textbf{Description} \\
\midrule
\multicolumn{2}{l}{\textit{Node types}} \\
SOURCE & path source \\
SINK   & path sink \\
IPIN   & logic input pin \\
OPIN   & logic output pin \\
CPIN   & clock input pin \\
\addlinespace[2pt]
\midrule
\multicolumn{2}{l}{\textit{Edge types}} \\
PRIMITIVE\_COMBINATIONAL & combinational delay \\
PRIMITIVE\_CLOCK\_LAUNCH & clock-to-Q delay \\
PRIMITIVE\_CLOCK\_CAPTURE & setup time \\
INTERCONNECT & net delay \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Feature Engineering}

For each timing graph, we construct node and edge features to describe
topological structure, placement geometry, timing priors, and
routing-related context. Common scalar attributes are normalized before
being fed into the model, while categorical attributes are encoded by
one-hot representations. All density-related statistics are computed on
the pruned RRG introduced in Sec.~\ref{sec:rrg_pruning}, using the valid
node set $V_{\mathrm{rr}}^{\mathrm{valid}}$ and valid edge set
$E_{\mathrm{rr}}^{\mathrm{valid}}$. The complete node and edge features
are summarized in Table~\ref{tab:node_feat_timing} and
Table~\ref{tab:edge_feat_timing}.

\begin{table}[t]
\centering
\caption{Node features used in the timing graph.}
\label{tab:node_feat_timing}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lll@{}}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type}\\
\midrule
\multirow{2}{*}{Placement}
& x-coordinate
& float\\
& y-coordinate
& float\\
\midrule
\multirow{4}{*}{Topology}
& logic level
& integer\\
& global longest logic level
& integer\\
& node type
& one-hot\\
& fanin / fanout
& integer\\
\midrule
\multirow{1}{*}{Timing Prior}
& post-placement arrival-time prior
& float\\
\bottomrule
\end{tabular*}
\end{table}

\begin{table}[t]
\centering
\caption{Edge features used in the timing graph.}
\label{tab:edge_feat_timing}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lll@{}}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type}\\
\midrule
\multirow{2}{*}{Geometry}
& HPWL
& float\\
& Manhattan distance
& float\\
\midrule
\multirow{2}{*}{Bounding-box Density}
& node-density max / avg
& float\\
& edge-usage-density max / avg
& float\\
\bottomrule
\end{tabular*}
\end{table}

For each timing node \(v\in\mathcal{V}_{\mathrm{tg}}\), we compute its
logic level on the combinational subgraph as a node-level topological
feature. We further compute the maximum logic level of the whole graph
and broadcast it to all nodes as a global topological feature.

In addition, we extract a post-placement arrival-time prior from the
timing information generated by VTR after placement. Let
\begin{equation}
\begin{aligned}
t_{\max}^{\mathrm{pl}}
&=
\max_{u\in\mathcal{V}_{\mathrm{tg}}} t_u^{\mathrm{pl}},
\quad
\tilde{t}^{\mathrm{pl}}_v
&=
\frac{t^{\mathrm{pl}}_v}{t_{\max}^{\mathrm{pl}}}.
\end{aligned}
\label{eq:pl_prior}
\end{equation}
Here, \(t_{\max}^{\mathrm{pl}}\) denotes the maximum post-placement
arrival time of the same design.

For each edge \((u,v)\), let \(\mathcal{B}_{uv}\) denote the axis-aligned bounding box
defined by the placement coordinates of nodes $u$ and $v$.
We first map the placed timing
nodes to their corresponding RRG nodes and partition them according to
tile locations. For each tile \(b\in\mathcal{B}_{uv}\), the node density
is defined as
\begin{equation}
\rho^{\mathrm{node}}_b
=
\frac{N_b^{\mathrm{valid}}}{N_b^{\mathrm{all}}},
\end{equation}
where \(N_b^{\mathrm{valid}}\) is the number of valid source/sink RRG
nodes in tile \(b\), and \(N_b^{\mathrm{all}}\) is the total number of
source/sink RRG nodes in that tile. If the denominator is zero, the
corresponding density is set to zero. The maximum and average node
densities over \(\mathcal{B}_{uv}\) are then used as edge features.

Similarly, each tile accumulates one count for every net in the design 
whose bounding box covers that tile. Let \(C_b\) denote the accumulated bounding-box
count on tile \(b\), and let \(E_b^{\mathrm{valid}}\) denote the number
of valid routing edges in that tile. The edge-usage density is defined as
\begin{equation}
\rho^{\mathrm{edge}}_b
=
\frac{C_b}{E_b^{\mathrm{valid}}},
\end{equation}
where the density is set to zero if \(E_b^{\mathrm{valid}}=0\). The
maximum and average edge-usage densities over \(\mathcal{B}_{uv}\) are
also used as edge features.

\subsection{Timing Propagation Network}

We propose a \emph{Timing Propagation Network} (TPN) for pre-routing
timing estimation. It consists of
a heterogeneous message-passing stage, an edge-gated propagation
stage, and two prediction heads for node-level and graph-level outputs.
Let \(\mathcal{V}_{\mathrm{tg}}\) denote the set of timing nodes, and let
\(\mathcal{E}_{\mathrm{tg}}^{(k)}\) denote the directed timing-edge set of
relation type \(k\).

The first stage applies several heterogeneous message-passing layers to
learn local representations from multi-type timing edges. For each
timing node \(v\in\mathcal{V}_{\mathrm{tg}}\), let \(\mathbf{x}_v\) denote
its raw node feature. For each directed edge
\((u,v)\in\mathcal{E}_{\mathrm{tg}}^{(k)}\), let
\(\mathbf{a}_{uv}^{(k)}\) denote its raw edge feature. Raw node
features are projected by a single shared node MLP,
while raw edge
features are projected by relation-specific edge MLPs:
\begin{equation}
\mathbf{h}_v^{(0)}=\mathrm{MLP}_{\mathrm{node}}(\mathbf{x}_v),
\end{equation}
\begin{equation}
\mathbf{e}_{uv}^{(k)}=\mathrm{MLP}_{\mathrm{edge}}^{(k)}(\mathbf{a}_{uv}^{(k)}),
\end{equation}
where \(\mathrm{MLP}_{\mathrm{edge}}^{(k)}\) has independent parameters for
each edge type \(k\). 
The encoded node and edge features are then processed by
$L$ heterogeneous message-passing layers. For layer
$l=1,\dots,L$, the aggregated message for node $v$ from
edge type $k$ is computed as
\begin{equation}
\mathbf{m}_v^{(k,l)}
=
\max_{u\in\mathcal{N}_k(v)}
\mathrm{MSG}^{(k,l)}\!\left(\mathbf{h}_u^{(l-1)},\,\mathbf{e}_{uv}^{(k)}\right),
\label{eq:stage1_msg}
\end{equation}
and the node representation is updated as
\begin{equation}
\mathbf{h}_v^{(l)}
=
\mathrm{MLP}_{\mathrm{upd}}^{(l)}
\!\left(
\mathbf{h}_v^{(l-1)}
+\sum_{k}\mathbf{m}_v^{(k,l)}
\right),
\label{eq:stage1_upd}
\end{equation}
where $\mathcal{N}_k(v)$ denotes the predecessors of $v$
connected by edge type $k$, and $\mathrm{MSG}^{(k,l)}$ is
a relation-specific message function with independent
parameters per type. Max aggregation within each edge type
captures the dominant predecessor along each timing-arc
category, consistent with the max-plus semantics of STA.

The second stage performs edge-gated max propagation to model
long-range timing dependency. Its input consists of the node
representations \(\tilde{\mathbf{h}}_v^{(0)}\) and the encoded edge
features \(\mathbf{e}_{uv}^{(k)}\). For each directed edge \((u,v)\), we
compute a scalar gate from its edge representation using a shared
trainable linear layer:
\begin{equation}
w_{uv}=g(\mathbf{e}_{uv}^{(k)})=\sigma\!\left(\mathbf{W}_{g}\mathbf{e}_{uv}^{(k)}+b_{g}\right),
\end{equation}
where \(\mathbf{W}_{g}\) and \(b_{g}\) are learnable parameters, and
\(\sigma(\cdot)\) is the sigmoid function. The same gate \(w_{uv}\) is
reused across all propagation steps. This design treats the edge gate as
a fixed propagation prior derived from edge attributes, avoiding redundant computation
across steps.

Let \(\tilde{\mathbf{h}}_v^{(t)}\) denote the node representation at
propagation step \(t\). For \(t=1,\dots,T\), one propagation step is
defined as
\begin{equation}
\mathbf{m}_v^{(t)}
=
\max_{u\in\mathrm{Pred}(v)}
\left(
w_{uv}\,\tilde{\mathbf{h}}_u^{(t-1)}
\right),
\end{equation}
\begin{equation}
\tilde{\mathbf{h}}_v^{(t)}
=
\mathrm{Fuse}^{(t)}
\!\left(
\tilde{\mathbf{h}}_v^{(t-1)},\mathbf{m}_v^{(t)}
\right),
\end{equation}
where
\[
\mathrm{Pred}(v)=\{u\in\mathcal{V}_{\mathrm{tg}}\mid (u,v)\in\mathcal{E}_{\mathrm{tg}}^{(k)}\ \text{for some}\ k\}
\]
denotes the set of predecessors of \(v\) over all edge types. The
step-specific residual fusion module is formulated as
\begin{equation}
\mathrm{Fuse}^{(t)}(\mathbf{h},\mathbf{m})
=
\mathrm{LN}
\!\left(
\mathbf{h}
+
\mathrm{MLP}^{(t)}
\!\left(
[\mathbf{h}\Vert\mathbf{m}]
\right)
\right),
\end{equation}
where \(\mathrm{MLP}^{(t)}\) has independent learnable parameters at step
\(t\), \([\cdot\Vert\cdot]\) denotes concatenation, and \(\mathrm{LN}\)
denotes layer normalization. In each step, the source-node
representation is first scaled by the edge gate, then aggregated by
element-wise max over all incoming edges, and finally fused with the
previous node state through residual update and normalization.

After propagation, TPN produces both node-level and graph-level outputs.
For each node \(v\in\mathcal{V}_{\mathrm{tg}}\), the normalized arrival
time is predicted by a node-level MLP, denoted by \(\hat{y}_v\). For
graph-level prediction, we first apply global max pooling over the final
node embeddings to obtain a graph representation, and then use a
graph-level MLP to predict the normalized CPD, as formulated in
Eqs.~\eqref{hg} and~\eqref{yg}.
\begin{equation}
\mathbf{h}_{\mathcal{G}}
=
\max_{v\in\mathcal{V}_{\mathrm{tg}}} \tilde{\mathbf{h}}_v^{(T)},
\label{hg}
\end{equation}
\begin{equation}
\hat{y}_{\mathcal{G}}
=
\mathrm{MLP}_{\mathrm{graph\_out}}(\mathbf{h}_{\mathcal{G}}).
\label{yg}
\end{equation}

Both outputs are normalized by \(t_{\max}^{\mathrm{pl}}\), the maximum
post-placement arrival-time prior of the same graph. This normalization
reduces scale variation across circuits and prevents large designs from
dominating training.
We use
mean squared error (MSE) to supervise both prediction tasks jointly:
\begin{equation}
\mathcal{L}
=
\frac{1}{|\mathcal{V}_{\mathrm{tg}}|}
\sum_{v\in\mathcal{V}_{\mathrm{tg}}}
\left(
\hat{y}_v-\frac{t_v}{t_{\max}^{\mathrm{pl}}}
\right)^2
+
\alpha
\left(
\hat{y}_{\mathcal{G}}-\frac{\mathrm{CPD}}{t_{\max}^{\mathrm{pl}}}
\right)^2.
\label{loss}
\end{equation}
where \(t_v\) denotes the target arrival time of node \(v\), and
\(\alpha>0\) balances the two loss terms. During inference, the final
arrival time and CPD are recovered by rescaling the predictions with
\(t_{\max}^{\mathrm{pl}}\).



\section{Experiments}
\label{exp}



\subsection{Experimental Setup}

We construct the dataset from 48 benchmark circuits drawn from the VTR8~\cite{vtr9}
and MCNC~\cite{mcnc} benchmark suites. For each design instance, the
post-placement timing graph is used as the model input, and the
supervision is extracted from the final post-routing timing analysis in
VPR, including node-level arrival times and the CPD.

To ensure sufficient diversity, the dataset is generated along three
variation axes: resource variation, placement variation, and circuit
variation. The corresponding configuration space is summarized in
Table~\ref{tab:config_space}. In total, the dataset contains 70000+ valid design instances.

We evaluate the proposed model under three generalization protocols:
cross-circuit, cross-architecture, and
cross-channel-width generalization. For cross-circuit generalization,
the 48 circuits are split into 24/7/17 for
training/validation/testing. For cross-architecture and cross-channel-width
generalization, the five architecture variants and the five
channel-width settings are partitioned into 3/1/1 groups. In each 
protocol, the test set is disjoint from the training set along the
corresponding variation axis, while the remaining axes are fully
covered.

The proposed model is implemented in PyTorch~\cite{paszke2019pytorch} and PyG~\cite{pyg}, and trained using
Adam with a learning rate of \(10^{-3}\) for 100 epochs. Model training
is performed on a server equipped with NVIDIA GeForce RTX 4090 GPUs.
All experiments are conducted under the
same environment for fair comparison.

\begin{table}[t]
\centering
\scriptsize
\caption{Configuration space used for dataset generation.}
\label{tab:config_space}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{p{1.7cm} p{2.0cm} p{4.5cm}}
\toprule
\textbf{Variation Axis} & \textbf{Factor} & \textbf{Values} \\
\midrule

\multirow{2}{*}{Resource}
& Architecture
& 5 VTR architectures: baseline, depop50, htree0, htree0+routedCLK, and htree0short \\

& Channel width
& \{50, 80, 110, 150, 200\} \\

\midrule

\multirow{4}{*}{Placement}
& Seed
& \{1, 2, 3, 4\} \\

& Inner\_num
& \{0.3, 0.5, 0.7, 0.9\} \\

& Fix\_pins
& \{free, random\} \\

& Place\_algorithm
& \{criticality\_timing, slack\_timing\} \\

\midrule

Circuit
& Benchmarks
& 48 circuits from the VTR8 and MCNC benchmark suites \\

\bottomrule
\end{tabular}
\end{table}

\subsection{Baselines and Model Configuration}

We compare the proposed model against four representative GNN
backbones, including GCN~\cite{kipf2017semi},
GIN~\cite{xu2019powerful},
GraphSAGE~\cite{hamilton2017inductive}, and
GAT~\cite{velickovic2018graph}.

For the proposed model, the propagation depth is set to \(T=9\), the
number of heterogeneous message-passing layers is set to \(L=3\), and the hidden
dimension is set to 128 for all layers. For fair comparison, all generic GNN baselines are configured with 9 layers, the same hidden dimension of 128, and an identical graph-level prediction head.
Specifically, GAT uses 4 attention heads, GraphSAGE uses the mean
aggregator, and GCN and GIN are both equipped with residual
connections. All models are trained using Adam with a learning rate of
\(10^{-3}\) for 100 epochs, and the final model is selected based on
the validation loss. ReLU is used as the activation function, no
dropout is applied, and the loss coefficient \(\alpha\) in
Eq.~(\ref{loss}) is set to 1.0.



\subsection{Evaluation Metrics}
We evaluate the predictor from both value accuracy and ranking quality.
Given the ground-truth value $y_i$ and the predicted value $\hat{y}_i$ for sample $i$, we first use mean absolute percentage error (MAPE) to measure relative prediction error:
\begin{equation}
\mathrm{MAPE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|,
\label{eq:mape}
\end{equation}
where $N$ is the number of evaluated samples.
A smaller MAPE indicates more accurate value prediction.
For ease of interpretation, we also report the corresponding accuracy as
\begin{equation}
\mathrm{Accuracy}=1-\mathrm{MAPE},
\label{eq:acc}
\end{equation}
where MAPE is expressed in decimal form.
Equivalently, when MAPE is reported as a percentage in tables, the accuracy can be written as $100\%-\mathrm{MAPE}(\%)$.
A larger accuracy indicates better prediction quality.

For ranking quality, we use Spearman's rank correlation coefficient:
\begin{equation}
\rho
=
1-\frac{6\sum_{i=1}^{N} d_i^2}{N(N^2-1)},
\label{eq:spearman}
\end{equation}
where $d_i$ is the rank difference between the ground-truth and predicted values of sample $i$.
A larger $\rho$ indicates better consistency between the predicted ranking and the true ranking, which is important for identifying critical endpoints.

We further use Top-10 overlap to evaluate whether the predictor can recover the truly most critical endpoints:
\begin{equation}
\mathrm{Top\mbox{-}10\ Overlap}
=
\frac{
\left|
\mathcal{T}_{\mathrm{pred}}^{10}
\cap
\mathcal{T}_{\mathrm{true}}^{10}
\right|
}{10}\times 100\%,
\label{eq:top10}
\end{equation}
where $\mathcal{T}_{\mathrm{pred}}^{10}$ and $\mathcal{T}_{\mathrm{true}}^{10}$ denote the sets of the predicted and ground-truth top-10 most critical endpoints, respectively.
A larger Top-10 overlap means better critical-path identification capability.

In addition, we report the coefficient of determination:
\begin{equation}
R^2
=
1-
\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{N}(y_i-\bar{y})^2},
\label{eq:r2}
\end{equation}
where $\bar{y}$ is the mean of the ground-truth values.
A larger $R^2$ indicates better overall regression quality.


\begin{table*}[t]
\centering
\scriptsize
\caption{Comparison of different methods under circuit-level, cross-architecture, and cross-channel-width evaluation settings. Best results in each row are highlighted in bold.}
\label{tab:main_results}
\setlength{\tabcolsep}{2.8pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lVcccVcccVcccVcccVccc}
\toprule
\multirow{2}{*}{\textbf{Circuit}} &
\multicolumn{3}{cV}{\textbf{GCN}} &
\multicolumn{3}{cV}{\textbf{GIN}} &
\multicolumn{3}{cV}{\textbf{GraphSAGE}} &
\multicolumn{3}{cV}{\textbf{GAT}} &
\multicolumn{3}{c}{\textbf{Ours}} \\
\cmidrule(lr){2-4}
\cmidrule(lr){5-7}
\cmidrule(lr){8-10}
\cmidrule(lr){11-13}
\cmidrule(lr){14-16}
& \(R^2\) & MAPE (\%) & \(\rho\)
& \(R^2\) & MAPE (\%) & \(\rho\)
& \(R^2\) & MAPE (\%) & \(\rho\)
& \(R^2\) & MAPE (\%) & \(\rho\)
& \(R^2\) & MAPE (\%) & \(\rho\) \\
\midrule

\multicolumn{16}{c}{\textbf{Circuit-level test results}} \\
\midrule
bgm              & \textbf{0.94} & 46.37 & 0.94 & 0.86 & 55.95 & 0.94 & 0.93 & 50.42 & \textbf{0.95} & \textbf{0.94} & 49.43 & 0.91 & 0.93 & \textbf{16.47} & 0.93 \\
blob\_merge      & 0.86 &  9.61 & 0.86 & 0.87 &  9.35 & 0.86 & 0.85 & 10.21 & 0.84 & 0.84 & 10.42 & 0.84 & \textbf{0.91} & \textbf{5.39} & \textbf{0.91} \\
boundtop         & 0.86 & 10.15 & \textbf{0.88} & \textbf{0.87} &  9.35 & \textbf{0.88} & 0.85 & 10.01 & 0.87 & \textbf{0.87} &  9.76 & \textbf{0.88} & 0.85 & \textbf{6.71} & 0.87 \\
ch\_intrinsics   & 0.83 & 10.23 & 0.91 & 0.86 &  9.06 & 0.91 & 0.84 &  9.92 & 0.90 & 0.85 &  9.25 & \textbf{0.92} & \textbf{0.87} & \textbf{5.99} & \textbf{0.92} \\
diffeq1          & 0.96 & 12.71 & 0.88 & 0.94 & 12.39 & 0.87 & 0.92 & 16.00 & 0.85 & 0.92 & 14.83 & 0.85 & \textbf{0.99} & \textbf{9.93} & \textbf{0.92} \\
diffeq2          & \textbf{0.99} & 14.92 & \textbf{0.92} & \textbf{0.99} & 14.19 & \textbf{0.92} & \textbf{0.99} & 15.22 & \textbf{0.92} & \textbf{0.99} & 14.36 & \textbf{0.92} & \textbf{0.99} & \textbf{11.59} & \textbf{0.92} \\
mcml             & \textbf{0.99} & 10.31 & 0.91 & \textbf{0.99} & 13.88 & 0.90 & \textbf{0.99} & 11.23 & 0.90 & \textbf{0.99} & 11.28 & 0.90 & \textbf{0.99} & \textbf{3.61} & \textbf{0.92} \\
mkDelayWorker32B & 0.72 & 13.99 & \textbf{0.89} & 0.71 & 13.92 & 0.88 & 0.73 & 13.60 & 0.88 & \textbf{0.74} & 13.57 & \textbf{0.89} & 0.72 & \textbf{9.33} & 0.88 \\
mkPktMerge       & \textbf{0.90} & 10.14 & \textbf{0.85} & \textbf{0.90} &  9.96 & \textbf{0.85} & 0.89 & 11.16 & 0.84 & 0.89 & 10.35 & \textbf{0.85} & 0.88 & \textbf{7.56} & 0.84 \\
mkSMAdapter4B    & 0.94 & 11.65 & \textbf{0.97} & \textbf{0.95} & 10.77 & \textbf{0.97} & 0.94 & 12.05 & 0.96 & 0.94 & 11.91 & 0.96 & \textbf{0.95} & \textbf{7.57} & \textbf{0.97} \\
or1200           & 0.96 &  6.24 & 0.97 & 0.96 &  6.22 & 0.97 & 0.95 &  6.82 & 0.97 & 0.96 &  6.40 & 0.97 & \textbf{0.97} & \textbf{3.67} & \textbf{0.98} \\
raygentop        & 0.80 & 11.21 & 0.90 & 0.80 & 10.55 & 0.89 & 0.81 & 12.10 & 0.89 & 0.81 & 12.87 & 0.90 & \textbf{0.84} & \textbf{8.12} & \textbf{0.91} \\
sha              & 0.84 & 10.68 & 0.87 & \textbf{0.85} &  9.64 & 0.88 & 0.83 & 11.07 & 0.87 & 0.84 & 11.11 & 0.87 & 0.80 & \textbf{6.97} & \textbf{0.92} \\
s38417           & 0.67 & 20.64 & 0.81 & 0.68 & 16.64 & 0.81 & 0.75 & 25.33 & 0.87 & 0.75 & 21.77 & 0.85 & \textbf{0.86} & \textbf{9.11} & \textbf{0.90} \\
frisc            & 0.89 & 16.78 & 0.95 & 0.81 & 22.00 & 0.91 & 0.83 & 21.33 & 0.93 & 0.81 & 19.44 & 0.92 & \textbf{0.92} & \textbf{8.12} & \textbf{0.97} \\
dsip             & 0.89 & 13.42 & 0.85 & 0.90 & 13.11 & 0.83 & 0.74 & 19.56 & \textbf{0.89} & \textbf{0.94} & \textbf{9.06} & \textbf{0.89} & 0.86 & 11.55 & 0.85 \\
tseng            & 0.55 & 37.36 & 0.80 & 0.66 & 29.11 & 0.83 & 0.74 & 28.63 & 0.85 & 0.68 & 33.37 & 0.88 & \textbf{0.85} & \textbf{12.52} & \textbf{0.91} \\
\midrule
Average          & 0.85 & 15.67 & 0.89 & 0.86 & 15.65 & 0.89 & 0.86 & 16.74 & 0.89 & 0.87 & 15.83 & 0.89 & \textbf{0.89} & \textbf{8.48} & \textbf{0.91} \\

\specialrule{0.9pt}{2pt}{2pt}
\multicolumn{16}{c}{\textbf{Cross-architecture test results}} \\
\midrule
Test set average  & 0.87 & 12.74 & 0.90 & \textbf{0.88} & 12.28 & 0.90 & 0.87 & 13.55 & 0.89 & 0.87 & 12.81 & 0.90 & \textbf{0.88} & \textbf{8.51} & \textbf{0.91} \\

\specialrule{0.9pt}{2pt}{2pt}
\multicolumn{16}{c}{\textbf{Cross-channel-width test results}} \\
\midrule
Test set average  & 0.85 & 13.52 & 0.90 & 0.86 & 12.96 & 0.89 & 0.86 & 13.74 & 0.89 & 0.86 & 13.08 & 0.89 & \textbf{0.89} & \textbf{10.01} & \textbf{0.91} \\
\bottomrule
\end{tabular}%
}
\end{table*}

\begin{table}[t]
\centering
\scriptsize
\caption{Comparison of Top-10 overlap and CPD MAPE under different evaluation settings, including generic GNN baselines and ablated variants of the proposed model.}
\label{tab:overlap_mape}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{l|ccc|ccc}
\toprule
\multirow{2}{*}{\textbf{Method}} &
\multicolumn{3}{c|}{\textbf{Top-10 overlap}} &
\multicolumn{3}{c}{\textbf{CPD MAPE}} \\
\cmidrule(lr){2-4}
\cmidrule(lr){5-7}
& Circuit & Channel & Arch & Circuit & Channel & Arch \\
\midrule
GCN       & 0.47 & 0.49 & 0.48 & 12.04 & 15.78 & 16.98 \\
GIN       & 0.45 & 0.52 & 0.55 & 42.85 & 38.78 & 45.32 \\
GraphSAGE & 0.52 & 0.59 & 0.54 & 20.27 & 19.44 & 18.35 \\
GAT       & 0.49 & 0.52 & 0.54 & 12.47 & 13.47 & 15.56 \\
RRG-GAE~\cite{dai2025rrg_gae}& 0.34& 0.51& 0.50& 31.77& 24.77&21.83\\
\midrule
w/o Stage II & 0.53 & 0.64 & 0.54 & 17.88 & 16.15 & 17.14 \\
w/o Stage I  & 0.57 & 0.68& \textbf{0.60} & 40.16 & 39.58 & 40.62 \\
\midrule
Ours      & \textbf{0.58} & \textbf{0.70} & \textbf{0.60} & \textbf{8.28} & \textbf{7.55} & \textbf{8.39} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Results and Analysis}
Table~\ref{tab:main_results} reports the comparison with four generic
GNN baselines under circuit-level, cross-architecture, and
cross-channel-width evaluation settings. Overall, the proposed model
achieves the best average performance under the circuit-level setting,
with \(R^2=0.89\), MAPE \(=8.48\%\), and
Spearman's \(\rho=0.91\). Compared with the strongest generic baseline
in this setting, the proposed model substantially reduces the node-level
arrival-time prediction error while also improving the ranking-based
metrics, indicating that it provides more accurate node-level arrival-time
prediction and better preserves the relative ordering of timing-critical
nodes.

At the circuit level, the proposed model shows particularly clear
advantages on several designs with relatively challenging timing
distributions, such as \textit{bgm}, \textit{mcml}, \textit{or1200},
\textit{frisc}, and \textit{tseng}, where the MAPE is consistently
lower than that of the generic GNN baselines. This suggests that the
proposed propagation design is more effective at capturing long-range
timing dependencies and timing criticality. At the same time, the
proposed model is not uniformly best on every individual circuit,
which indicates that different circuits may favor different inductive
biases; however, its superior average results demonstrate better overall
robustness across diverse designs.

Under the cross-architecture setting, the proposed model continues to
show strong generalization ability. It achieves the best or tied-best
\(R^2\), the lowest MAPE, and the best ranking consistency among all
compared methods. This result indicates that the learned timing
representation transfers well across different FPGA architecture
variants, rather than overfitting to a single resource organization.

Under the cross-channel-width setting, the proposed model again achieves
the best MAPE and the best ranking metrics, but its \(R^2\) is higher
than those of the generic GNN baselines. This suggests that when the
routing-resource budget changes significantly, the proposed model still
preserves the relative ordering of timing-critical nodes well, but the
absolute calibration of predicted arrival times becomes more difficult.
In other words, the model remains reliable for node ranking, while its
value regression is more sensitive to distribution shift in routing
capacity.

Table~\ref{tab:overlap_mape} further compares the endpoint-level
Top-10 overlap and the graph-level CPD MAPE, and also includes two
ablated variants of the proposed model. Among all generic GNN baselines,
the proposed model achieves the highest Top-10 overlap in all three
settings and consistently attains the lowest CPD MAPE, demonstrating
its advantage in both critical-endpoint identification and graph-level
delay estimation.

In Table~\ref{tab:overlap_mape}, the variant without Stage I
(\textit{w/o Stage I}) directly uses the raw node features as the input
to Stage II, while the variant without Stage II (\textit{w/o Stage II})
removes the edge-gated max-propagation stage and directly applies
the prediction heads after Stage I.

The ablation results further verify the effectiveness of both stages.
Removing Stage II consistently degrades both Top-10 overlap and CPD
MAPE under all three settings, showing that this stage is important for
long-range timing propagation. In contrast, removing Stage I causes an
even larger degradation in CPD prediction, although the Top-10 overlap
remains relatively competitive. This indicates that Stage I is mainly
responsible for learning effective local heterogeneous timing
representations, while Stage II further propagates critical timing
information over long ranges. Overall, the two stages are complementary,
and combining them yields the most balanced performance across different
evaluation settings.

\begin{table*}[t]
\centering
\renewcommand{\arraystretch}{0.9}
\caption{Performance Comparison of Our Method against Baseline and ABC Retiming on the Test Dataset.}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{c|ccc|cccc|cccc}
\toprule
\multirow{2}{*}{\textbf{Benchmark}} & \multicolumn{3}{c|}{\textbf{Baseline}} & \multicolumn{4}{c|}{\textbf{ABC Retiming}} & \multicolumn{4}{c}{\textbf{Our method}} \\
& \#Node & \#Level & $F_\text{max}$(MHz) & \#Node & \#Level & $F_\text{max}$(MHz) & Impr. & \#Node & \#Level & $F_\text{max}$(MHz) & Impr. \\
\midrule
bgm              & 76213 & 95  & 58.41  & -     & -   & -      & -      & 76673 & 87  & 60.07  & 2.85\% \\
blob\_merge      & 40081 & 102 & 93.29  & 40081 & 102 & 96.46  & 3.41\% & 40089 & 95  & 104.34 & 11.84\% \\
boundtop         & 376   & 9   & 456.73 & 376   & 9   & 446.63 & -2.21\%& 378   & 9   & 455.15 & -0.34\% \\
ch\_intrinsics   & 346   & 15  & 412.43 & 346   & 15  & 452.85 & 9.80\% & 346   & 12  & 437.29 & 6.03\% \\
diffeq1          & 2749  & 77  & 40.20  & 2749  & 75  & 124.19 & 208.96\% & 2749  & 75  & 124.19 & 208.96\% \\
diffeq2          & 2196  & 75  & 54.73  & 2203  & 65  & 135.45 & 147.47\% & 2198  & 69  & 116.69 & 113.19\% \\
mcml             & 407351 & 397 & 14.46 & -     & -   & -      & -      & 407397 & 397 & 20.09  & 38.88\% \\
mkDelayWorker32B & 1362  & 21  & 125.32 & 1362  & 21  & 76.41  & -39.03\% & 1362  & 21  & 76.41  & -39.03\% \\
mkPktMerge       & 550   & 5   & 226.65 & 550   & 5   & 313.09 & 38.14\% & 550   & 5   & 285.05 & 25.76\% \\
mkSMAdapter4B    & 4890  & 35  & 166.94 & 4885  & 35  & 211.73 & 26.84\% & 4898  & 35  & 223.12 & 33.65\% \\
or1200           & 12833 & 148 & 68.46  & 12878 & 143 & 76.98  & 12.44\% & 12845 & 148 & 73.92  & 7.97\% \\
raygentop        & 4321  & 41  & 165.49 & -     & -   & -      & -      & 4325  & 30  & 254.79 & 53.96\% \\
sha              & 11793 & 205 & 70.21  & 11835 & 98  & 75.77  & 7.90\% & 11815 & 94  & 76.73  & 9.28\% \\
 s38417& 13870& 28& 121.34& 13885& 37& 107.42& -11.47\%& 13885& 29& 130.64&7.67\%\\
frisc& 8011  & 66  & 75.08  & 8011  & 66  & 72.63  & -3.26\% & 8015  & 61  & 85.92  & 14.44\% \\
dsip             & 3176  & 10  & 345.68 & 3176  & 10  & 362.09 & 4.75\% & 3176  & 8   & 424.67 & 22.85\% \\
tseng            & 2501  & 45  & 147.46 & 2501  & 45  & 139.82 & -5.18\% & 2506  & 39  & 156.76 & 6.30\% \\
\midrule
\textbf{Average} & \multicolumn{3}{c|}{--} & \multicolumn{3}{c}{--} & \textbf{26.57\%}& \multicolumn{3}{c}{--} & \textbf{30.83\%}\\
\bottomrule
\end{tabular}
\label{tab:retime}
\vspace{2pt}
\end{table*}


\section{Application: Predictor-Guided Retiming}
           
To demonstrate the practical utility of the proposed timing predictor, we apply it to guide iterative retiming for FPGA timing optimization. Conventional retiming in ABC~\cite{abc} uses a unit-delay model and optimization decisions are made primarily according to logic depth. Although this abstraction enables efficient heuristic search, it often fails to capture the true timing bottlenecks of FPGA designs, whose critical paths are strongly affected by placement, interconnect delay, and routing resource constraints. As a result, the logic-critical path identified by ABC may deviate substantially from the physically critical path after implementation, leading to suboptimal retiming decisions.

Recent work has started to incorporate physical timing feedback into retiming. In particular, Zhu \textit{et al.}~\cite{ref_iterret} proposed an iterative and verifiable retiming framework that extracts critical-path information from STA reports and uses it to guide subsequent retiming steps. While this improves timing awareness compared with pure logic-level retiming, it still relies on explicit STA feedback from downstream implementation stages to reveal physical criticality. In contrast, our framework introduces a learned physical-aware timing predictor directly into the retiming loop, enabling timing-guided optimization already at the post-placement stage.

\begin{figure}[t]
    \centering
    \begin{subfigure}{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figure/backward.drawio.pdf}
        \caption{Backward retiming.}
        \label{fig:retiming_backward}
    \end{subfigure}
    %\vspace{0.5pt}
    \begin{subfigure}{0.49\linewidth}
        \centering
        \includegraphics[width=\linewidth]{figure/forward.drawio.pdf}
        \caption{Forward retiming.}
        \label{fig:retiming_forward}
    \end{subfigure}

    \caption{Two fundamental retiming moves in ABC.}
    \label{fig:retiming_moves}
\end{figure}

Fig.~\ref{fig:retiming_moves} illustrates the forward and backward retiming transformations considered in ABC. Based on these primitive moves, our method replaces the original unit-delay guidance with predictor-estimated timing criticality, as summarized in Algorithm~\ref{alg:gnn_retime_final}. Starting from the current netlist \(N_{opt}\), the predictor first identifies the top-10 critical paths of the design. The retiming engine then evaluates feasible forward and backward moves only for nodes located on these predicted critical paths, computes the gain of each candidate transformation, and applies the one with the largest benefit. Importantly, after each retiming move, the predictor is re-invoked on the updated netlist so that the critical-path set can be refreshed dynamically as the circuit structure evolves. This process repeats until no profitable move can be found.


\makeatletter
\newcommand{\COMMENT}[1]{\STATE \textcolor{blue}{// #1}}
\makeatother
\begin{algorithm}[!t]
\caption{Timing Predictor-enhanced Retiming within ABC Framework}
\label{alg:gnn_retime_final}
\footnotesize
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
    \State \textbf{return} $N_{opt}$
\end{algorithmic}
\end{algorithm}

Unlike iterative flows that repeatedly invoke downstream timing analysis during optimization, our method uses the predictor as a lightweight timing oracle inside the retiming loop, while real routing STA is performed only once after retiming converges for final evaluation. In this way, the method preserves the efficiency of heuristic retiming while substantially improving the physical relevance of the optimization objective. Since the optimization is performed at the post-placement stage, the runtime is dominated by placement rather than routing, and the cost of final routing is negligible in our evaluation. 

Table~\ref{tab:retime} compares the baseline flow, standard ABC retiming, and our predictor-guided retiming method. Here, \#Node and \#Level denote the number of AIG nodes and the logic depth reported by ABC, respectively. Overall, our method achieves an average \(F_{\max}\) improvement of \(30.83\%\) over the baseline, outperforming the \(26.57\%\) average improvement of standard ABC retiming. These results indicate that replacing logic-depth guidance with physical-aware critical-path prediction leads to higher-quality retiming decisions. The advantage is especially pronounced on routing-dominated designs such as \textit{dsip}, \textit{raygentop}, and \textit{blob\_merge}, where long interconnect paths are the primary performance limiters. In contrast, for circuits whose critical paths are more logic-dominated, such as \textit{diffeq2}, the improvement is more limited, since reducing physical path delay is less effective than directly optimizing logic depth.

Table~\ref{tab:retiming_comparison} further compares our approach with prior works. Unlike conventional ABC retiming, which relies on logic depth under a unit-delay model, and unlike prior iterative retiming approaches that depend on STA-derived critical-path feedback, our method integrates predicted timing information directly into the optimization loop at the post-placement stage. This use case highlights that the proposed predictor serves not only as a timing analysis model, but also as an optimization primitive embedded in the CAD flow, enabling physically informed retiming decisions earlier and more efficiently.




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
~\cite{b6}    & ASIC       & \XSolidBrush                                                & \XSolidBrush                                             & Gate-Level                                               & STA                    \\
\cite{b3}               & FPGA       & \XSolidBrush                                                & \XSolidBrush                                             & Post Routing                                             & Simple Timing Model    \\
RTA~\cite{ref:rta}    & ASIC       & \XSolidBrush                                                & \XSolidBrush                                             & Post Placement                                           & Rough Estimation       \\
Ours                  & FPGA       & \CheckmarkBold                                              & \CheckmarkBold                                           & Post Placement                                                 & Predicted Timing Info. \\ \bottomrule
\end{tabular}}
\vspace{3pt}
\end{table}




\section{Conclusion}

FPGA pre-routing timing estimation remains challenging because final timing is jointly determined by logic structure, placement, and routing resources, while the exact routing information is still unavailable before detailed routing.

In this paper, we presented PASTE, a physical-aware surrogate for FPGA pre-routing timing estimation. PASTE operates on the post-placement timing graph through a heterogeneous GNN formulation, so that both timing dependencies and node types can be naturally modeled. To better capture routing-induced delay variation, we further introduced design-aware RRG pruning together with physically informative features, including node density and edge usage density. In addition, the proposed TPN adopts a two-stage propagation scheme: local heterogeneous message passing with max aggregation, followed by long-range edge-gated max propagation, which explicitly aligns with the max-plus semantics of static timing analysis.

Experimental results show that PASTE achieves a node-level arrival-time MAPE of 8.48\% under the circuit-level setting, while requiring less than one second of inference time after placement. As a downstream application, PASTE-guided retiming improves $F_{\max}$ by 30.83\%, outperforming ABC retiming, which achieves 26.57\%.

These results demonstrate that a physically informed learned surrogate can serve as an effective timing oracle in FPGA CAD flows, enabling timing-driven optimization before routing. Future work includes extending PASTE to multi-clock designs and integrating the predictor into placement refinement.



\bibliographystyle{IEEEtran}
\bibliography{DAC-GNNRet}






\end{document}
\endinput
%%
%% End of file `sample-sigconf.tex'.
