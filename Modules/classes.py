import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from copy import deepcopy
from scipy.linalg import expm
from tqdm import tqdm, trange
import networkx as nx
from networkx import Graph

#%%

class Hamiltonian:
    """
    A class representing a Hamiltonian for quantum systems.

    Attributes
    ----------
    matrix : ndarray
        The Hamiltonian matrix.
    dim : int
        The dimension of the Hamiltonian matrix.
    energies : ndarray
        The energies corresponding to the diagonal elements of the Hamiltonian matrix.
    graph : Network
        A visual representation of the Hamiltonian matrix as a graph.
    couplings : ndarray
        The off-diagonal elements of the Hamiltonian matrix representing couplings.
    energy_diff_mat : ndarray
        Matrix of energy differences between eigenvalues.
    probabilities : ndarray
        Probabilities computed from the Hamiltonian matrix.

    Methods
    -------
    get_random_weak_ham(dim, emin, emax, tol)
        Generates a random weak Hamiltonian matrix.
    get_graph(col_map_nodes="hsv", menu_toggle=False, color_font="white", bgcolor="#222222", vmin=0, vmax=1)
        Generates a graph representation of the Hamiltonian matrix.
    get_probabilities(hbar=1, tol=1e-15)
        Calculates transition probabilities based on the Hamiltonian matrix.
    """
    def __init__(self, matrix = None, dim = 4, hbar = 1, emin = 0, emax = 10,
                col_map_nodes = "hsv", min_node_alpha = 0.1, min_edge_alpha = 0.25):
        """
        Initializes the Hamiltonian with either a provided matrix or a randomly generated weak Hamiltonian.

        Parameters
        ----------
        matrix : ndarray, optional
            The Hamiltonian matrix. If None, a random matrix is generated.
        dim : int, optional
            The dimension of the Hamiltonian matrix.
        hbar : float, optional
            Reduced Planck constant.
        emin : float, optional
            Minimum energy level for random generation.
        emax : float, optional
            Maximum energy level for random generation.
        """
        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = self.get_random_ham(dim = dim, emin = emin, emax = emax)
        self.dim = self.matrix.shape[0]
        self.energies =  self.matrix.diagonal()
        
        self.couplings = self.matrix - np.diag(self.energies)
        self.energy_diff_mat = (self.energies[:,None] - self.energies[None,:])
        self.max_abs_coupling = np.abs(self.couplings.max())
        self.min_abs_coupling = np.abs(self.couplings.min())
        
        self.graph = self.get_graph(col_map_nodes = col_map_nodes, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha)

    def get_random_ham(self, dim, emin, emax):
        """
        Generates a random weak Hamiltonian matrix.

        Parameters
        ----------
        dim : int
            The dimension of the Hamiltonian matrix.
        emin : float
            Minimum energy level for random generation.
        emax : float
            Maximum energy level for random generation.

        Returns
        -------
        ndarray
            A randomly generated weak Hamiltonian matrix.
        """
        choices_nr = 1000 * dim
        pssbl_energies = np.linspace(emin, emax, choices_nr)
        energies = np.random.choice(pssbl_energies, size = dim)
        e_diffs = np.abs(energies[:, None] - energies[None, :])
        diag_idx = np.diag_indices_from(e_diffs)
        pssbl_offs = np.linspace(-e_diffs.max(), e_diffs.max(), choices_nr)
        tmp = np.random.choice(pssbl_offs, size = (dim, dim))
        tmp[diag_idx] = energies
        H_mat = tmp + tmp.conj().T
        return H_mat
    
    def get_graph(self,
                 col_map_nodes = "hsv", 
                 min_node_alpha = 0.1, min_edge_alpha = 0.25):
        """
        Generates a graph representation of the state vector.

        Parameters
        ----------
        H : Hamiltonian
            The Hamiltonian governing the system's evolution.
        networkx : bool, optional
            Whether to use NetworkX for graph representation.
        col_map_nodes : str, optional
            Color map for the nodes in the graph.

        Returns
        -------
        Network or Graph
            A graph representation of Hamiltonian.
        """
        
        nodes = np.arange(self.dim) # list of nodes IDs
        coups_idx = np.nonzero(self.couplings * np.tri(self.dim, self.dim, -1).T) # upper off-diagonal elements of hamiltonian
        edges = [(coups_idx[0][i], coups_idx[1][i]) for i in range(len(coups_idx[0]))] # edge tuples

        graph = Graph(edges) # initializing edges
        graph.add_nodes_from(nodes) # ensuring that nodes without edges are included
        nodes_colors = [list(coloring(self.energies[node], vmin =  self.energies.min(), vmax =  self.energies.max(), col_map = col_map_nodes)) for node in nodes]

        for node in nodes:
            alpha_val = normalize_val(self.energies[node], vmin = self.energies.min(), vmax = self.energies.max())
            graph.nodes[node]["color"] = to_hex(nodes_colors[node])
            graph.nodes[node]["label"] = node
            graph.nodes[node]["value"] = self.energies[node]
            graph.nodes[node]["alpha"] = alpha_val if alpha_val >= min_node_alpha else min_node_alpha
        for edge in edges:
            alpha_val = normalize_val(np.abs(self.matrix[edge]), vmin = 0, vmax =  np.abs(self.energy_diff_mat[edge]))
            graph.edges[edge]["alpha"] = alpha_val if alpha_val >= min_edge_alpha else min_edge_alpha
        
        return graph
    
    def show_graph(self, layout_idx = 0, plt_style = "classic", figure_size_list = [7.50, 7.50], plt_auto_layout = True,
                   usetex = True, font_family = 'serif', weight = 'normal', font_size = 18,
                   cmap_nodes = 'coolwarm', cmap_edges = "hsv", fig_face_color = "black", ax_face_color = "black", edge_constant_color = "white"
                  ):
        plt.style.use(plt_style)
        plt.rc('text', usetex = usetex)
        plt.rc('font', family = font_family, weight = weight)
        plt.rc('font', size = font_size)
        plt.rcParams["figure.figsize"] = figure_size_list
        plt.rcParams["figure.autolayout"] = plt_auto_layout
        
        fig, ax = plt.subplots()
        fig.set_facecolor(fig_face_color)
        ax.set_facecolor(ax_face_color)
        
        G = self.graph
            
        layouts = [nx.circular_layout, 
                   nx.spring_layout, 
                   nx.spectral_layout,
                   nx.kamada_kawai_layout,
                   nx.planar_layout,
                   nx.spring_layout,
                   nx.shell_layout,
                   nx.random_layout]
        
        layout = layouts[layout_idx](G)
        
        nx.draw_networkx_edges(G, layout, edgelist = dict(G.edges).keys(),
                       alpha = [G.edges[edge]["alpha"] for edge in dict(G.edges).keys()],
                       edge_color = edge_constant_color)
        nx.draw_networkx_nodes(G, layout,
                               #alpha = [G.nodes[node]["alpha"] for node in dict(G.nodes).keys()],
                               node_color = [G.nodes[node]["color"] for node in dict(G.nodes).keys()])
        nx.draw_networkx_labels(G,layout)


class State:
    """
    A class representing a quantum state.

    Attributes
    ----------
    state : ndarray
        The state vector.
    graph : Network or Graph
        A visual representation of the state vector.

    Methods
    -------
    get_graph(H, networkx=False, col_map_nodes="hsv")
        Generates a graph representation of the state vector.
    move(U, col_map_nodes="hsv", networkx=False)
        Updates the state vector by applying a unitary transformation.
    propagate(H, dt=0.1, t_final=None, networkx=False, sim_tol=1e-15, col_map_nodes="hsv")
        Propagates the state vector over time.
    """
    def __init__(self, state_v, H,
                 col_map_nodes = "coolwarm", col_map_edges = "hsv", sim_tol = 1e-15,
                 min_node_alpha = 0.1, min_edge_alpha = 0.25):
        """
        Initializes the State with a given state vector and Hamiltonian.

        Parameters
        ----------
        state_v : ndarray
            The initial state vector.
        H : Hamiltonian
            The Hamiltonian governing the system's evolution.
        networkx : bool, optional
            Whether to use NetworkX for graph representation.
        col_map_nodes : str, optional
            Color map for the nodes in the graph.
        sim_tol : float, optional
            Tolerance level for similarity calculations during propagation.
        """
        self.state = state_v # state vector
        self.dense = state_v[:,None] @ state_v[None,:].conj() # density matrix 
        self.state_graph = self.get_state_graph(H, col_map_nodes = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # graph of vector state
        self.dense_graph = self.get_dense_graph(H, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # graph of density matrix

    def get_state_graph(self, H, 
                        col_map_nodes = "hsv", 
                        min_node_alpha = 0.1, min_edge_alpha = 0.25):
        """
        Generates a graph representation of the state vector.

        Parameters
        ----------
        H : Hamiltonian
            The Hamiltonian governing the system's evolution.
        networkx : bool, optional
            Whether to use NetworkX for graph representation.
        col_map_nodes : str, optional
            Color map for the nodes in the graph.

        Returns
        -------
        Network or Graph
            A graph representation of the state vector.
        """
        probs = np.abs(self.state) ** 2
        phase = np.angle(self.state)
        matrix = H.matrix.copy()
        diagonal_idx = np.diag_indices_from(matrix)
        matrix[diagonal_idx] = probs
        
        
        nodes = np.arange(H.dim)
        coups_idx = np.nonzero(H.couplings * np.tri(H.dim, H.dim, -1).T)
        edges = [(coups_idx[0][i], coups_idx[1][i]) for i in range(len(coups_idx[0]))]

        graph = Graph(edges)
        graph.add_nodes_from(nodes)
        rgba_colors = [list(coloring(phase[node], vmin = -np.pi, vmax = np.pi, col_map = col_map_nodes)) for node in nodes]

        for node in nodes:
            alpha_val = normalize_val(probs[node], vmin = 0, vmax = 1)
            rgba_colors[node][-1] =  s_profile(matrix[node, node]).real
            graph.nodes[node]["color"] = to_hex(rgba_colors[node])
            graph.nodes[node]["label"] = node
            graph.nodes[node]["value"] = self.state[node]
            graph.nodes[node]["alpha"] = alpha_val if alpha_val >= min_node_alpha else min_node_alpha
        for edge in edges:
            alpha_val = normalize_val(np.abs(H.matrix[edge]), vmin = 0, vmax =  np.abs(H.energy_diff_mat[edge]))
            graph.edges[edge]["alpha"] = alpha_val if alpha_val >= min_edge_alpha else min_edge_alpha

        return graph

    def get_dense_graph(self, H, 
                        col_map_nodes = "coolwarm", col_map_edges = "hsv",
                        min_node_alpha = 0.1, min_edge_alpha = 0.25):
        """
        Generates a graph representation of the state density matrix.

        Parameters
        ----------
        H : Hamiltonian
            The Hamiltonian governing the system's evolution.
        networkx : bool, optional
            Whether to use NetworkX for graph representation.
        col_map_nodes : str, optional
            Color map for the edges in the graph.
        Returns
        -------
        Network or Graph
            A graph representation of the state density matrix.
        """
        lower_triangle_remover = np.tri(H.dim, H.dim, -1).T
        phases = np.angle(self.dense) # matrix of phases of density matrix off diagonals
        
        
        nodes = np.arange(H.dim)
        upper_triangle = phases * lower_triangle_remover
        phases_idx = np.nonzero(upper_triangle)
        edges = [(phases_idx[0][i], phases_idx[1][i]) for i in range(len(phases_idx[0]))]
        graph = Graph(edges)
        graph.add_nodes_from(nodes)

        upper_triangle_abs_sqrd = np.abs(upper_triangle)**2
        for idx, edge in enumerate(edges):
            alpha_val = normalize_val(upper_triangle_abs_sqrd[edge], vmin = 0, vmax = upper_triangle_abs_sqrd.max().real) # 0 <= |alpha_ij|^2 <= p_ii*p_jj
            graph.edges[edge]["color"] = to_hex(coloring(phases[edge], vmin = -np.pi, vmax = np.pi, col_map = col_map_edges))
            graph.edges[edge]["alpha"] = alpha_val if alpha_val >= min_edge_alpha else min_edge_alpha 
        for node in nodes:
            alpha_val = normalize_val(self.dense[node, node], vmin = 0, vmax = 1)
            graph.nodes[node]["label"] = node
            graph.nodes[node]["value"] = self.dense[node, node]
            graph.nodes[node]["color"] = to_hex(coloring(self.dense[node, node].real, vmin = 0, vmax = 1, col_map = col_map_nodes))
            graph.nodes[node]["alpha"] = alpha_val.real if alpha_val.real >= min_node_alpha else min_node_alpha

        return graph
    
    def show_graph(self, state = True,
                   layout_idx = 0, plt_style = "classic", figure_size_list = [7.50, 7.50], plt_auto_layout = True,
                   usetex = True, font_family = 'serif', weight = 'normal', font_size = 18,
                   cmap_nodes = 'coolwarm', cmap_edges = "hsv", fig_face_color = "black", ax_face_color = "black", edge_constant_color = "white"
                  ):
        #plt.style.use(plt_style)
        plt.rc('text', usetex = usetex)
        plt.rc('font', family = font_family, weight = weight)
        plt.rc('font', size = font_size)
        plt.rcParams["figure.figsize"] = figure_size_list
        plt.rcParams["figure.autolayout"] = plt_auto_layout
        
        fig, ax = plt.subplots()
        fig.set_facecolor(fig_face_color)
        ax.set_facecolor(ax_face_color)
        
        if state:
            G = self.state_graph
        else:
            G = self.dense_graph
            
        layouts = [nx.circular_layout, 
                   nx.spring_layout, 
                   nx.spectral_layout,
                   nx.kamada_kawai_layout,
                   nx.planar_layout,
                   nx.spring_layout,
                   nx.shell_layout,
                   nx.random_layout]
        
        layout = layouts[layout_idx](G)
        
        if state:
            nx.draw_networkx_edges(G, layout, edgelist = dict(G.edges).keys(),
                           alpha = [G.edges[edge]["alpha"] for edge in dict(G.edges).keys()],
                           edge_color = edge_constant_color)
            nx.draw_networkx_nodes(G, layout,
                                   alpha = [G.nodes[node]["alpha"] for node in dict(G.nodes).keys()],
                                   node_color = [G.nodes[node]["color"] for node in dict(G.nodes).keys()])
            nx.draw_networkx_labels(G,layout)
        else:
            nx.draw_networkx_edges(G, layout, edgelist = dict(G.edges).keys(), 
                                   alpha = [G.edges[edge]["alpha"] for edge in dict(G.edges).keys()],
                                   edge_color = [G.edges[edge]["color"] for edge in dict(G.edges).keys()],
                                   edge_cmap = plt.get_cmap(cmap_edges),
                                   edge_vmin = -np.pi, edge_vmax = np.pi)
            nx.draw_networkx_nodes(G, layout,
                                   node_color = [G.nodes[node]["color"] for node in dict(G.nodes).keys()],
                                   #node_color = "white",
                                   #alpha = [G.nodes[node]["alpha"] for node in dict(G.nodes).keys()]
                                  )
            nx.draw_networkx_labels(G,layout)
        

    def move(self, H, U, col_map_edges = "hsv", col_map_nodes = "coolwarm",
             min_node_alpha = 0.1, min_edge_alpha = 0.25):
        #self.state = np.dot(self.state, H.probabilities)
        self.state = U @ self.state
        self.state = self.state / np.linalg.norm(self.state)
        self.dense = self.state[:, None] @ self.state[None,:].conj()
        self.state_graph = self.get_state_graph(H, col_map_nodes = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # in this case color of nodes is complex phase of edges
        self.dense_graph = self.get_dense_graph(H, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha)

    def propagate(self, H, 
                  dt = None, t_final = None, 
                  sim_tol = 1e-15, 
                  col_map_edges = "hsv", col_map_nodes = "coolwarm",
                  min_node_alpha = 0.1, min_edge_alpha = 0.25):
        states = [deepcopy(self)] # initializing list
        vect_distance = lambda v, u :  np.dot(v,u) / (np.linalg.norm(v)*np.linalg.norm(u)) # cos similarity function
        U = expm(-1j * H.matrix * dt)
        
        if t_final == None:
            self.move(H, U, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # move initial state
            init_sim = vect_distance(self.state, states[0].state) # initial similarity to initialize while loop
            similarities = [0.1, init_sim] # first entry is introduced to initialize while loop
            idx = 0 # dummy index
            
            pbar = tqdm(total = idx+1) # progress bar
            while np.abs(similarities[-1] - similarities[-2]) > sim_tol: # looping until similarity is stable
                state.move(U, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # move state
                states.append(deepcopy(self)) # append copy of new state
                similarities.append(vect_distance(self.state, states[idx].state)) # append similarity with previous state
                idx += 1 # updating idx
                pbar.update(1)
            return states, similarities
        steps = int(t_final / dt)
        for _ in trange(steps, desc = "Propagating"):
            self.move(H, U, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges, min_node_alpha = min_node_alpha, min_edge_alpha = min_edge_alpha) # move state
            states.append(deepcopy(self)) # store new state
        return states
        
def s_profile(x, k=10, x0=0.5):
    # Logistic function
    g = lambda x: 1 / (1 + np.exp(-k * (x - x0)))
    
    # Normalizing to ensure f(0) = 0 and f(1) = 1
    g0 = g(0)
    g1 = g(1)
    f = lambda x: (g(x) - g0) / (g1 - g0)
    
    return f(x)

def normalize_val(val, vmin, vmax):
    norm = plt.Normalize(vmin = vmin, vmax = vmax) # normalizes value between bounds
    norm_value = lambda val : min(max(val, vmin), vmax) # constraining value between two bounds
    normalized_value = lambda val: norm(norm_value(val)) # normalizing
    return normalized_value(val)

def coloring(val, vmin, vmax, col_map):
    cmap = plt.get_cmap(col_map) # obtaining color map for nodes 
    rgba_color = lambda val: cmap(normalize_val(val, vmin = vmin, vmax = vmax)) # getting rgba
    return rgba_color(val)
