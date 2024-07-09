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
        self.probabilities = self.get_probabilities(hbar = hbar, tol = tol)

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
        tmp = 0.025 / e_diffs # the 0.025 makes it so that H_i,j = |H_ii-H_jj| / 10
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
    
        tmp = self.matrix * np.tri(self.dim, self.dim, -1) # returns H where all diagonal and upper triangle are set to 0
        where_non_zero = np.nonzero(tmp) # returns tuple of arrays which indicate where non zero elements appear
    
        graph.add_edges([(int(where_non_zero[0][i]), int(where_non_zero[1][i])) for i in range(len(where_non_zero[0]))])
        return graph
        
    def get_probabilities(self, hbar=1, tol=1e-15):
        """
        Calculates transition probabilities based on the Hamiltonian matrix.

        Parameters
        ----------
        hbar : float, optional
            Reduced Planck constant.
        tol : float, optional
            Tolerance level for calculations.

        Returns
        -------
        ndarray
            Complex probabilities computed from the Hamiltonian matrix.
        """
        # Compute density of states as 1 / (energy difference + eps) to avoid division by zero
        #density = (self.state[:,None] - self.state[None,:]) / (H.energy_diff_mat + tol)# density of states
        density = 1.0 / (np.abs(self.energy_diff_mat) + tol)
        
        # Compute unnormalized probabilities using Fermi's golden rule
        probabilities = (2 * np.pi / hbar) * density * np.abs(self.couplings) ** 2
        
        # Normalize the probabilities
        #norms = np.sqrt(np.sum(np.abs(probabilities)**2, axis = 0)) # normalization
        #normalized_probabilities = probabilities / norms[:,None]
        norms = np.sum(probabilities, axis=1, keepdims=True)
        normalized_probabilities = probabilities / norms

        # Convert to complex probabilities with phases
        complex_probabilities = np.sqrt(normalized_probabilities) * np.exp(1j * np.angle(self.couplings))

        return complex_probabilities # normalized_probabilities

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
                 networkx = False, col_map_nodes = "hsv",
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
        self.state = state_v 
        self.graph = self.get_graph(H, networkx = networkx, tol = tol, col_map_nodes = col_map_nodes)

    def get_graph(self, H, 
                  networkx = False, col_map_nodes = "hsv",
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
            # Coloring
            
            norm = plt.Normalize(vmin = -np.pi, vmax = np.pi) # normalizes value between bounds
            norm_value = lambda val : min(max(val, -np.pi), np.pi) # constraining value between two bounds
            normalized_value = lambda val: norm(norm_value(val)) # normalizing
            
            cmap = plt.get_cmap(col_map_nodes) # obtaining color map for nodes 
            rgba_color = lambda val: cmap(normalized_value(val)) # getting rgba            
            
            # Graph
            coups_idx = np.nonzero(H.couplings * np.tri(H.dim, H.dim, -1))
            edges = [(coups_idx[0][i], coups_idx[1][i]) for i in range(len(coups_idx[0]))]
            graph = Graph(edges)
            rgba_colors = [list(rgba_color(phase[node].real)) for node in range(H.dim)]
            

            for node in range(H.dim):
                rgba_colors[node][-1] =  s_profile(matrix[node, node]).real
                graph.nodes[node]["color"] = to_hex(rgba_colors[node])
                graph.nodes[node]["label"] = node
                graph.nodes[node]["value"] = self.state[node]
        else: 
            graph = Hamiltonian(matrix).get_graph()
            
        return graph

    def move(self, H, U, col_map_nodes = "hsv", networkx = False):
        #self.state = np.dot(self.state, H.probabilities)
        self.state = U @ self.state
        self.state = self.state / np.linalg.norm(self.state)
        self.graph = self.get_graph(H, networkx = networkx, col_map_nodes = col_map_nodes)

    def propagate(self, H, 
                  dt = None, t_final = None, 
                  networkx = False, sim_tol = 1e-15, 
                  col_map_nodes = "hsv"):
        states = [deepcopy(self)] # initializing list
        vect_distance = lambda v, u :  np.dot(v,u) / (np.linalg.norm(v)*np.linalg.norm(u)) # cos similarity function
        U = expm(-1j * H.matrix * dt)
        
        if t_final == None:
            self.move(H, U, networkx = networkx, col_map_nodes = col_map_nodes) # move initial state
            init_sim = vect_distance(self.state, states[0].state) # initial similarity to initialize while loop
            similarities = [0.1, init_sim] # first entry is introduced to initialize while loop
            idx = 0 # dummy index
            
            pbar = tqdm(total = idx+1) # progress bar
            while np.abs(similarities[-1] - similarities[-2]) > sim_tol: # looping until similarity is stable
                state.move(U, networkx = True, col_map_nodes = col_map_nodes) # move state
                states.append(deepcopy(self)) # append copy of new state
                similarities.append(vect_distance(self.state, states[idx].state)) # append similarity with previous state
                idx += 1 # updating idx
                pbar.update(1)
            return states, similarities
        steps = int(t_final / dt)
        for _ in trange(steps, desc = "Propagating"):
            self.move(H, U, networkx = True, col_map_nodes = col_map_nodes) # move state
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
