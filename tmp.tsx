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

\algrenewcommand\alglinenumber[1]{\scriptsize #1:}

\pagestyle{empty}
\setlength{\textfloatsep}{3pt plus 1pt minus 1pt}
\setlength{\floatsep}{6pt plus 1pt minus 1pt}
\setlength{\intextsep}{6pt plus 1pt minus 1pt}

\renewcommand{\arraystretch}{1}
\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}


\begin{document}

\title{PASTE: A \underline{P}hysical-\underline{A}ware \underline{S}urrogate for FPGA Pre-Routing \underline{T}iming \underline{E}stimation、、g}
\maketitle

\begin{abstract}



Accurate pre-routing timing estimation is a fundamental challenge in FPGA design, since the final path delay is jointly determined by logic structure, placement, routing-resource constraints, and congestion. Existing early-stage methods often rely on coarse proxies or simplified delay models, which limits their usefulness for node-level timing analysis and critical-path identification before detailed routing.
In this paper, we present PASTE, a physical-aware surrogate for FPGA pre-routing timing estimation. PASTE first extracts physically informative features from the timing graph of a placed design, including placement-derived geometry, routing-related attributes, and congestion-aware signals, so as to capture routing-induced timing variation beyond pure logic topology. Based on these features, we further propose a novel Timing Propagation Network (TPN), a directed edge-aware graph neural network, to predict node-level normalized arrival times and graph-level critical path delay.
Implemented in VTR, PASTE achieves an $R^2$ of 0.98 and a Spearman correlation of 0.98 on the test set, while requiring only millisecond-level inference time, which is substantially smaller than the runtime of actual routing and static timing analysis. As a downstream validation case, we further integrate PASTE into a predictor-guided retiming flow, which improves $F_{\max}$ by 8\% on average over logic-only ABC retiming. These results demonstrate that physical-aware graph-based timing estimation can serve as an effective surrogate for early-stage timing analysis and timing-driven FPGA optimization.

\end{abstract}


\begin{IEEEkeywords}
GNN, STA, Timing Prediction
\end{IEEEkeywords}








\section{Introduction}

Field-programmable gate arrays (FPGAs) are widely used for hardware
acceleration due to their flexibility and strong domain-specific
performance. As FPGA designs continue to scale, timing closure has
become a major challenge in the CAD flow. Many timing-driven
optimizations, such as logic restructuring, placement refinement, and
sequential optimization, must be performed before detailed routing.
Therefore, accurate pre-routing timing prediction is highly desirable,
since it can provide early timing guidance without incurring the cost of
iterative routing.

Pre-routing timing prediction in FPGAs is challenging because timing is
jointly determined by logic delay, placement, and the
architecture-constrained routing fabric. In particular, routing delay
can contribute a substantial portion of the critical-path delay in
modern FPGAs~\cite{ref:route_delay}, making logic-only estimation
insufficient for identifying true timing bottlenecks. As summarized in
Table~\ref{tab:related}, most prior FPGA learning-based methods focus on
indirect objectives or restricted timing targets, rather than directly
predicting arrival time for timing analysis. A recent work by Dai
\emph{et al.}~\cite{dai2025rrg_gae} took an important step toward FPGA
pre-routing arrival-time prediction. However, its explicit all-net
modeling can still incur considerable overhead on large-scale circuits,
and its training objective is not fully end-to-end on arrival time.

In this work, we propose a design-aware pre-routing arrival-time
prediction framework for FPGA designs based on the post-placement timing
graph. Our method directly models timing dependency at the register
level and incorporates placement-derived and architecture-aware physical
features, enabling the predictor to capture both circuit structure and
FPGA-specific timing characteristics before detailed routing. Compared
with prior FPGA timing predictors, our framework directly targets
arrival time while avoiding the heavy cost of explicit all-net
modeling, making it more practical and scalable for large designs.
Experimental results show that the proposed predictor achieves accurate
pre-routing arrival-time estimation across diverse circuits and
implementation settings. As a case study, we further demonstrate its
effectiveness as a timing surrogate for downstream applications such as
physical-aware retiming.

The main contributions of this work are summarized as follows:
\begin{itemize}
    \item We formulate FPGA pre-routing timing prediction as a
    design-aware arrival-time estimation problem on the post-placement
    timing graph, targeting a timing quantity directly relevant to
    register-level performance analysis.

    \item We propose an FPGA-specific prediction framework that combines
    graph-structured timing dependency with placement-derived and
    architecture-aware physical features for accurate pre-routing timing
    modeling.

    \item We design the framework to learn arrival time in an end-to-end
    fashion without relying on explicit all-net modeling, improving its
    scalability to large circuits.

    \item We demonstrate that the proposed predictor generalizes across
    diverse circuits and implementation settings, and can serve as an
    effective timing surrogate for downstream timing-driven applications
    such as physical-aware retiming.
\end{itemize}

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
\cite{dai2025rrg_gae} & FPGA & Register-level & \CheckmarkBold & \textbf{Arrival Time} \\
Ours & FPGA & Register-level & \CheckmarkBold & \textbf{Arrival Time} \\
\bottomrule
\end{tabular}%
}
\end{table}


\section{Background}



\subsection{Pre-Routing Timing Estimation in FPGA Design}

Existing FPGA pre-routing timing estimation methods can be divided into three categories. The first category predicts indirect routing-related proxies from placement-derived features. Representative examples include wirelength prediction~\cite{wirelength}, congestion estimation~\cite{bbcong,conghls}, routability prediction~\cite{dl_routability,routenet}, and MUX-usage modeling~\cite{fGREP,sta_rout_mux}. Although useful for global quality assessment, these methods do not directly provide node-level timing information for critical-path analysis.

The second category predicts delay for specific targets. Hu \emph{et al.}~\cite{lutdelay} use machine learning to estimate delay during FPGA technology mapping. Ustun \emph{et al.}~\cite{hlspd} use graph neural networks to predict operation-level delay in FPGA HLS. Although effective in their target scenarios, these methods are tied to specific abstraction levels or design stages, and are therefore insufficient for general node-level or path-level timing estimation on placed netlists.

The third category uses heuristic routing-cost estimation in FPGA CAD tools. For example, VTR uses architecture-aware routing lookahead to estimate delay and congestion during maze routing~\cite{air,vtr9}. These estimates are designed to guide routing search rather than to serve as accurate timing surrogates for node-level arrival-time prediction or critical-path analysis. Moreover, their congestion estimates are heuristic and do not directly model the actual timing variation introduced by final routing choices.

Overall, existing FPGA methods either predict coarse proxies, address restricted delay subproblems, or provide routing heuristics. They do not provide general register-level timing prediction on placed netlists, which limits their ability to identify critical paths before detailed routing.

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
whose delay mainly depends on placement and wire parasitics. In
contrast, an FPGA net is implemented by selecting a legal path through
the fixed routing architecture, as illustrated in
Fig.~\ref{fig:rrg}. The routing architecture is typically modeled as a
routing resource graph (RRG), where nodes correspond to routing
resources or pins and edges denote programmable connections. As a result, 
its delay is strongly influenced by switch-box topology and
local congestion, rather than by geometric distance alone. Based on the RRG, a timing graph can be further constructed for timing analysis, as illustrated in Fig.~2(c). As can be observed from the timing graph, the total delay of a complete timing path typically consists of three components: the clock-to-Q delay, the net delay, and the combinational logic delay.

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
\mathcal{G}=(\mathcal{V},\mathcal{E}),
\)
where each timing node (denoted as \texttt{tnode} in VTR) is represented
as a graph node, and each timing edge (denoted as \texttt{tedge} in VTR)
is represented as a directed edge
describing a valid timing dependency between two timing nodes. This
timing graph is maintained and updated throughout the CAD flow as timing
information becomes more accurate. In particular, after placement, VTR
provides a timing graph whose edge delays already incorporate
logic-delay information and whose nodes carry placement-stage timing
annotations. These post-placement timing quantities provide useful prior
knowledge before detailed routing. In this work, we perform pre-routing
timing prediction on this post-placement timing graph. Node and edge features 
are then constructed on this post-placement timing graph.



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
\begin{tabular}{lll}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type/Range} \\
\midrule
\multirow{2}{*}{Placement}
& x-coordinate
& float / normalized
\\
& y-coordinate
& float / normalized
\\
\midrule
\multirow{4}{*}{Topology}
& logic level
& integer
\\
& global longest logic level
& integer / broadcast
\\
& node type
& categorical / one-hot
\\
& fanin / fanout
& integer
\\
\midrule
\multirow{1}{*}{Timing Prior}
& post-placement arrival-time prior
& float / normalized
\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Edge features used in the timing graph.}
\label{tab:edge_feat_timing}
\footnotesize
\begin{tabular}{lll}
\toprule
\textbf{Feature Group} & \textbf{Feature} & \textbf{Type/Range} \\
\midrule
\multirow{2}{*}{Geometry}
& HPWL
& float / normalized
\\
& Manhattan distance
& float / normalized
\\
\midrule
\multirow{2}{*}{Bounding-box Density}
& node-density max / avg
& float / normalized
\\
& edge-usage-density max / avg
& float / normalized
\\
\bottomrule
\end{tabular}
\end{table}

For each timing node \(v\), we compute its logic level $l_v$ on the
combinational subgraph as:
\begin{equation}
l_v=\max_{p\in\mathcal{P}(v)} |p|,
\end{equation}
where $\mathcal{P}(v)$ denotes the set of topological paths ending at
$v$. We further compute the maximum logic level $l_{\max}$ of the whole graph,
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
source/sink RRG nodes in that tile. Specifically, if the denominator is zero, the 
corresponding density is set to zero.The maximum and average values
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
where the density is set to zero if $E^{\mathrm{valid}}_b=0$.
Its maximum and average values over $\mathcal{B}_{uv}$ are used as
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

We propose a \emph{Timing Propagation Network} (TPN) for pre-routing
timing estimation. As illustrated in Fig.~\ref{fig:tpn}, it consists of
a heterogeneous message-passing stage, a delay-weighted propagation
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
features are first projected by a shared node MLP, while raw edge
features are projected by relation-specific edge MLPs:
\begin{equation}
\mathbf{h}_v^{(0)}=\mathrm{MLP}_{\mathrm{node}}(\mathbf{x}_v),
\end{equation}
\begin{equation}
\mathbf{e}_{uv}^{(k)}=\mathrm{MLP}_{\mathrm{edge}}^{(k)}(\mathbf{a}_{uv}^{(k)}),
\end{equation}
where \(\mathrm{MLP}_{\mathrm{edge}}^{(k)}\) has independent parameters for
each edge type \(k\). The encoded node and edge features are then
processed by \(L\) heterogeneous message-passing layers with sum
aggregation, which captures the combined contributions of multiple local
predecessors. The resulting node representations are denoted by
\(\tilde{\mathbf{h}}_v^{(0)}\). In contrast, Stage II uses max
aggregation to model the worst-predecessor behavior in timing
propagation.

The second stage performs delay-weighted max propagation to model
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
reused across all propagation steps. This design uses the fixed physical
delay attributes of each edge as a shared propagation prior, while also
avoiding redundant gate computation across steps.

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

Both outputs are normalized by the maximum post-placement arrival-time
prior of the same graph,
\(t_{\max}^{\mathrm{pl}}=\max_{u\in\mathcal{V}_{\mathrm{tg}}} t_u^{\mathrm{pl}}\),
where \(t_u^{\mathrm{pl}}\) denotes the post-placement arrival-time prior
of timing node \(u\). This normalization reduces scale variation across
circuits and prevents large designs from dominating training. We use
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
\right)^2,
\label{loss}
\end{equation}
where \(t_v\) denotes the target arrival time of node \(v\), and
\(\alpha>0\) balances the two loss terms. During inference, the final
arrival time and CPD are recovered by rescaling the predictions with
\(t_{\max}^{\mathrm{pl}}\).



\section{Experiments}
\label{exp}



\subsection{Experimental Setup}

We construct the dataset from 45 benchmark circuits drawn from the VTR8
and MCNC benchmark suites. For each design instance, the post-placement
timing graph is used as the model input, while the supervision is
obtained from the final post-routing timing analysis in VPR, including
node-level arrival times and the graph-level critical path delay (CPD).

To ensure sufficient diversity, the dataset is generated along three
variation axes: resource variation, placement variation, and circuit
variation. The corresponding configuration space is summarized in
Table~\ref{tab:config_space}.

We evaluate the proposed model under three generalization protocols:
cross-circuit generalization, cross-architecture generalization, and
cross-channel-width generalization. For cross-circuit generalization,
the 45 circuits are split into 26/5/19 for
training/validation/testing. For cross-architecture generalization, the
five architecture variants are partitioned into 3/1/1 groups. For
cross-channel-width generalization, the five channel-width settings are
also partitioned into 3/1/1 groups. In each protocol, the test set is
disjoint from the training set along the corresponding variation axis.
The remaining variation axes are fully covered in the training,
validation, and test sets unless otherwise specified.

The proposed model is implemented in PyTorch and DGL, and trained using
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
& seed
& \{1, 2, 3, 4\} \\

& inner\_num
& \{0.3, 0.5, 0.7, 0.9\} \\

& fix\_pins
& \{free, random\} \\

& place\_algorithm
& \{criticality\_timing, slack\_timing\} \\

\midrule

Circuit
& Benchmarks
& 45 circuits from the VTR8 and MCNC benchmark suites \\

\bottomrule
\end{tabular}
\end{table}

\subsection{Baselines, Ablation, and Model Configuration}

We compare the proposed model against four representative GNN
backbones, including GCN~\cite{kipf2017semi},
GIN~\cite{xu2019powerful},
GraphSAGE~\cite{hamilton2017inductive}, and
GAT~\cite{velickovic2018graph}. In addition, we include
RRG-GAE~\cite{dai2025rrg_gae}, a recent FPGA-specific pre-routing
arrival-time prediction framework, as a task-specific baseline. For
RRG-GAE, we directly adopt the original configuration reported in its
paper.

For the proposed model, the propagation depth is set to \(T=9\), the
number of propagation blocks is set to \(L=3\), and the hidden
dimension is set to 128 for all layers. For fair comparison, all
generic GNN baselines, including GCN, GIN, GraphSAGE, and GAT, are
configured with 9 layers and the same hidden dimension of 128.
Specifically, GAT uses 4 attention heads, GraphSAGE uses the mean
aggregator, and GCN and GIN are both equipped with residual
connections. These models use the same
activation function, optimizer, learning rate, number of training
epochs, and model-selection criterion as the proposed model.

To further analyze the contribution of the proposed design, we consider
two ablation variants. The first retains only the proposed
message-passing mechanism while removing the proposed propagation
structure. The second retains only the proposed propagation structure
while replacing the message-passing operator with a standard
GCN layer~\cite{kipf2017semi}. These ablations isolate the contributions
of the proposed message-passing design and propagation structure,
respectively.

All reimplemented models are trained using Adam with a learning rate of
\(10^{-3}\) for 100 epochs, and the final model is selected based on
the validation loss. ReLU is used as the activation function, and no
dropout is applied. For multi-task training, the loss coefficient
\(\alpha\) in Eq.~(\ref{loss}) is set to 1.0.





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





\section{Application: Predictor-Guided Retiming}

To demonstrate the practical utility of the proposed timing predictor, we apply it to guide iterative retiming for FPGA timing optimization. Conventional retiming in ABC~\cite{abc} is based on a unit-delay model, where each logic node is assigned an identical delay and optimization decisions are made primarily according to logic depth. Although this abstraction enables efficient heuristic search, it often fails to capture the true timing bottlenecks of FPGA designs, whose critical paths are strongly affected by placement, interconnect delay, and routing resource constraints. As a result, the logic-critical path identified by ABC may deviate substantially from the physically critical path after implementation, leading to suboptimal retiming decisions.

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

Fig.~\ref{fig:retiming_moves} illustrates the forward and backward retiming transformations considered in ABC. Based on these primitive moves, our method replaces the original unit-delay guidance with predictor-estimated timing criticality, as summarized in Algorithm~\ref{alg:gnn_retime_final}. Starting from the current netlist \(N_{opt}\), the predictor first identifies the top-$k$ critical paths of the design. The retiming engine then evaluates feasible forward and backward moves only for nodes located on these predicted critical paths, computes the gain of each candidate transformation, and applies the one with the largest benefit. Importantly, after each retiming move, the predictor is re-invoked on the updated netlist so that the critical-path set can be refreshed dynamically as the circuit structure evolves. This process repeats until no profitable move can be found.


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

Unlike iterative flows that repeatedly invoke downstream timing analysis during optimization, our method uses the predictor as a lightweight timing oracle inside the retiming loop, while real STA is performed only once after retiming converges for final evaluation. In this way, the method preserves the efficiency of heuristic retiming while substantially improving the physical relevance of the optimization objective. Since the optimization is performed at the post-placement stage, the runtime is dominated by placement rather than routing, and the cost of final routing is negligible in our evaluation. Even though the predictor is re-invoked after every retiming move, the overall end-to-end flow still achieves more than \(10\times\) average speedup over implementation-driven iterative alternatives.

Table~\ref{tab:retime} compares the baseline flow, standard ABC retiming, and our predictor-guided retiming method. Here, \#Node and \#Level denote the number of AIG nodes and the logic depth reported by ABC, respectively. Overall, our method achieves an average \(F_{\max}\) improvement of \(27.24\%\) over the baseline, outperforming the \(19.40\%\) average improvement of standard ABC retiming. These results indicate that replacing logic-depth guidance with physical-aware critical-path prediction leads to higher-quality retiming decisions. The advantage is especially pronounced on routing-dominated designs such as \textit{dsip}, \textit{raygentop}, and \textit{blob\_merge}, where long interconnect paths are the primary performance limiters. In contrast, for circuits whose critical paths are more logic-dominated, such as \textit{diffeq2}, the improvement is more limited, since reducing physical path delay is less effective than directly optimizing logic depth.

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
