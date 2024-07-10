import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pyvis.network import Network
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
    def __init__(self, matrix = None, dim = 4, hbar = 1, tol = 1e-15, emin = 0, emax = 10):
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
        tol : float, optional
            Tolerance level for calculations.
        emin : float, optional
            Minimum energy level for random generation.
        emax : float, optional
            Maximum energy level for random generation.
        """
        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = self.get_random_weak_ham(dim = dim, emin = emin, emax = emax, tol = tol)
        self.dim = self.matrix.shape[0]
        self.energies =  self.matrix.diagonal()
        self.graph = self.get_graph(vmin = self.energies.min(), vmax = self.energies.max())
        
        self.couplings = self.matrix - np.diag(self.energies)
        self.energy_diff_mat = (self.energies[:,None] - self.energies[None,:])
        self.max_abs_coupling = np.abs(self.couplings.max())
        self.min_abs_coupling = np.abs(self.couplings.min())

    def get_random_weak_ham(self, dim, emin, emax, tol):
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
        tol : float
            Tolerance level for calculations.

        Returns
        -------
        ndarray
            A randomly generated weak Hamiltonian matrix.
        """

        sorted_energies = np.sort(np.random.choice(np.linspace(emin, emax, 1000 * dim), size = dim, replace = True))
        e_diffs = np.abs(sorted_energies[:, None] - sorted_energies[None, :]) + tol
        diag_idx = np.diag_indices_from(e_diffs)
        tmp = 0.25 / e_diffs # the 0.025 makes it so that H_i,j = |H_ii-H_jj| / 1
        tmp[diag_idx] = sorted_energies
        H_mat = tmp + tmp.conj().T
        return H_mat

    def get_graph(self, col_map_nodes = "hsv",
                 menu_toggle = False, color_font = "white",
                 bgcolor="#222222", vmin = 0, vmax = 1):
        """
        Generates a graph representation of the Hamiltonian matrix.

        Parameters
        ----------
        col_map_nodes : str, optional
            Color map for the nodes.
        menu_toggle : bool, optional
            Toggle for displaying menu in the graph.
        color_font : str, optional
            Color of the font in the graph.
        bgcolor : str, optional
            Background color of the graph.
        vmin : float, optional
            Minimum value for normalization.
        vmax : float, optional
            Maximum value for normalization.

        Returns
        -------
        Network
            A graph representation of the Hamiltonian matrix.
        """
        # Coloring
        cmap = plt.get_cmap(col_map_nodes) # obtaining color map for nodes    
        
        norm_value = lambda val : min(max(val, vmin), vmax) # constraining value between two bounds
        norm = plt.Normalize(vmin = vmin, vmax = vmax) # normalizes value between bounds
        normalized_value = lambda val: norm(norm_value(val)) # normalizing
        
        rgba_color = lambda val: cmap(normalized_value(val)) # getting rgba
        hex_color = lambda val : to_hex(rgba_color(val)) # converting to hex
        
        # Graph
        graph = Network(bgcolor=bgcolor, font_color = color_font, select_menu = menu_toggle) # new empty graph
        nodes = [int(n) for n in range(self.dim)] # list of node IDs
        graph.add_nodes(nodes, 
                        color = [hex_color(en) for en in self.energies],
                        label = [f"{n}" for n in nodes]) # creating as many nodes as diagonal elements
    
        tmp = self.matrix * np.tri(self.dim, self.dim, -1).T # returns H where all diagonal and lower triangle are set to 0
        where_non_zero = np.nonzero(tmp) # returns tuple of arrays which indicate where non zero elements appear
    
        graph.add_edges([(int(where_non_zero[0][i]), int(where_non_zero[1][i])) for i in range(len(where_non_zero[0]))])
        return graph
     

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
    get_graph(H, networkx=False, col_map_nodes="hsv", tol=1e-15)
        Generates a graph representation of the state vector.
    move(U, col_map_nodes="hsv", networkx=False)
        Updates the state vector by applying a unitary transformation.
    propagate(H, dt=0.1, t_final=None, networkx=False, sim_tol=1e-15, col_map_nodes="hsv")
        Propagates the state vector over time.
    """
    def __init__(self, state_v, H,
                 networkx = True, 
                 col_map_nodes = "coolwarm", col_map_edges = "hsv",
                 tol = 1e-15, sim_tol = 1e-15):
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
        tol : float, optional
            Tolerance level for calculations.
        sim_tol : float, optional
            Tolerance level for similarity calculations during propagation.
        """
        self.state = state_v # state vector
        self.dense = state_v[:,None] @ state_v[None,:].conj() # density matrix 
        self.state_graph = self.get_state_graph(H, networkx = networkx, tol = tol, col_map_nodes = col_map_edges) # graph of vector state
        self.dense_graph = self.get_dense_graph(H, networkx = networkx, tol = tol, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges) # graph of density matrix

    def get_state_graph(self, H, 
                        networkx = True, col_map_nodes = "hsv",
                        tol = 1e-15):
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
        tol : float, optional
            Tolerance level for calculations.

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
        
        if networkx: # if we want graph representation to be handled by networkx
            nodes = np.arange(H.dim)
            coups_idx = np.nonzero(H.couplings * np.tri(H.dim, H.dim, -1).T)
            edges = [(coups_idx[0][i], coups_idx[1][i]) for i in range(len(coups_idx[0]))]
            
            graph = Graph(edges)
            graph.add_nodes_from(nodes)
            rgba_colors = [list(coloring(phase[node], vmin = -np.pi, vmax = np.pi, col_map = col_map_nodes)) for node in nodes]
            
            for node in nodes:
                #node = list(dict(graph.nodes).keys())[idx]
                #print(graph.nodes.values(), node)
                rgba_colors[node][-1] =  s_profile(matrix[node, node]).real
                graph.nodes[node]["color"] = to_hex(rgba_colors[node])
                graph.nodes[node]["label"] = node
                graph.nodes[node]["value"] = self.state[node]
            for edge in edges:
                alpha_val = normalize_val(np.abs(H.matrix[edge]), vmin = 0, vmax =  np.abs(H.energy_diff_mat[edge]))
                graph.edges[edge]["alpha"] = alpha_val if alpha_val >= 0.25 else 0.25
        else: 
            graph = Hamiltonian(matrix).get_graph()
            
        return graph
    
    def get_dense_graph(self, H, 
                        networkx = True, col_map_nodes = "coolwarm", col_map_edges = "hsv",
                        tol = 1e-15):
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
        tol : float, optional
            Tolerance level for calculations.

        Returns
        -------
        Network or Graph
            A graph representation of the state density matrix.
        """
        lower_triangle_remover = np.tri(H.dim, H.dim, -1).T
        phases = np.angle(self.dense) # matrix of phases of density matrix off diagonals
        
        if networkx: # if we want graph representation to be handled by networkx
            nodes = np.arange(H.dim)
            phases_idx = np.nonzero(phases * lower_triangle_remover)
            edges = [(phases_idx[0][i], phases_idx[1][i]) for i in range(len(phases_idx[0]))]
            graph = Graph(edges)
            graph.add_nodes_from(nodes)
            
            for idx, edge in enumerate(edges):
                alpha_val = normalize_val(np.abs(self.dense[edge])**2, vmin = 0, vmax = (self.dense[edge[0],edge[0]] * self.dense[edge[1],edge[1]]).real)
                graph.edges[edge]["color"] = to_hex(coloring(phases[edge], vmin = -np.pi, vmax = np.pi, col_map = col_map_edges))
                graph.edges[edge]["alpha"] = alpha_val if alpha_val >= 0.25 else 0.25 # 0 <= |alpha_ij|^2 <= p_ii*p_jj
            for node in nodes:
                graph.nodes[node]["label"] = node
                graph.nodes[node]["value"] = self.dense[node, node]
                graph.nodes[node]["color"] = to_hex(coloring(self.dense[node, node].real, vmin = 0, vmax = 1, col_map = col_map_nodes))
        else: 
            graph = Hamiltonian(matrix).get_graph()
            
        return graph

    def move(self, H, U, col_map_edges = "hsv", col_map_nodes = "coolwarm", networkx = True):
        #self.state = np.dot(self.state, H.probabilities)
        self.state = U @ self.state
        self.state = self.state / np.linalg.norm(self.state)
        self.dense = self.state[:, None] @ self.state[None,:].conj()
        self.state_graph = self.get_state_graph(H, networkx = networkx, col_map_nodes = col_map_edges) # in this case color of nodes is complex phase of edges
        self.dense_graph = self.get_dense_graph(H, networkx = networkx, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges)

    def propagate(self, H, 
                  dt = None, t_final = None, 
                  networkx = True, sim_tol = 1e-15, 
                  col_map_edges = "hsv", col_map_nodes = "coolwarm",):
        states = [deepcopy(self)] # initializing list
        vect_distance = lambda v, u :  np.dot(v,u) / (np.linalg.norm(v)*np.linalg.norm(u)) # cos similarity function
        U = expm(-1j * H.matrix * dt)
        
        if t_final == None:
            self.move(H, U, networkx = networkx, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges) # move initial state
            init_sim = vect_distance(self.state, states[0].state) # initial similarity to initialize while loop
            similarities = [0.1, init_sim] # first entry is introduced to initialize while loop
            idx = 0 # dummy index
            
            pbar = tqdm(total = idx+1) # progress bar
            while np.abs(similarities[-1] - similarities[-2]) > sim_tol: # looping until similarity is stable
                state.move(U, networkx = True, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges) # move state
                states.append(deepcopy(self)) # append copy of new state
                similarities.append(vect_distance(self.state, states[idx].state)) # append similarity with previous state
                idx += 1 # updating idx
                pbar.update(1)
            return states, similarities
        steps = int(t_final / dt)
        for _ in trange(steps, desc = "Propagating"):
            self.move(H, U, networkx = True, col_map_nodes = col_map_nodes, col_map_edges = col_map_edges) # move state
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
