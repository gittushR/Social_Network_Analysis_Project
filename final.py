from math import exp, sqrt, floor, log, factorial
import ndlib.models.epidemics as ep
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from matplotlib.colors import ListedColormap
from typing import List, Tuple

phi = {}


# Implementation of Algorithm 1 as Given in the paper
def estimate_diffusion_source_number(GI):
    # Step 1: Initialize core node set Vc
    Vc = set()

    # Step 2: Obtain the optimal impact factor sigma with potential entropy method
    sigma = 0.1

    # Step 3: Calculate the topological potential for each node in VI
    for vi in GI.nodes():
        phi[vi] = sum([GI.degree(vj) * exp(-(nx.shortest_path_length(GI, vi, vj) / sigma) ** 2) for vj in GI.nodes()])

    # Step 4: Find local maximum potential nodes in VI
    for vi in GI.nodes():
        if all(phi[vi] > phi[vj] for vj in GI.neighbors(vi)):
            Vc.add(vi)
    # print(Vc)
    # Step 5: Refine Vc using hop distance criteria
    # while True:
    #     found_swap = False
    #     for vi in Vc:
    #         for vj in Vc:
    #             if vi == vj:
    #                 continue
    #             if nx.shortest_path_length(GI, vi, vj) < floor(3 * sigma / sqrt(2)):
    #                 if phi[vi] >= phi[vj]:
    #                     Vc.discard(vj)
    #                     found_swap = True
    #                 else:
    #                     Vc.discard(vi)
    #                     found_swap = True
    #     if not found_swap:
    #         break

    # Step 6: Return the estimated diffusion source number k and the core node set Vc
    k = len(Vc)
    return k, Vc


def get_optimal_sigma(GI):
    # Initialize the set of candidate sigmas
    candidate_sigmas = [0.01 * rp for rp in range(1, 101)]

    # Calculate entropy for each sigma in candidate_sigmas
    entropy_values = []
    for sigma in candidate_sigmas:
        phi = {}
        for vi in GI.nodes():
            phi[vi] = sum([GI.degree(vj) * exp(-(nx.shortest_path_length(GI, vi, vj) / sigma) ** 2) for vj in GI.nodes()])
        entropy = -sum([phi[vi] * log(phi[vi]) for vi in GI.nodes() if phi[vi] != 0])
        entropy_values.append(entropy)

    # Find the sigma that minimizes entropy
    min_entropy = min(entropy_values)
    min_entropy_index = entropy_values.index(min_entropy)
    optimal_sigma = candidate_sigmas[min_entropy_index]
    return optimal_sigma


# Implementation of Algorithm 2 to get partition on the basis of peak and valley
def get_partition(GI, Vcc):
    partitions = []
    for i in Vcc:
        GIi = nx.Graph()
        vci = i
        GIi.add_node(vci, state=True)
        for vj in GI.neighbors(vci):
            if vj not in GIi.nodes:
                partition_node(vci, vj, GIi, GI)
        partitions.append(GIi)
    return partitions


def partition_node(vci, vj, GIi, GI):
    if all(phi[vj] <= phi[neighbors] for neighbors in GI.neighbors(vj)):
        GIi.add_node(vj, state=True)
        GIi.add_edge(vci, vj)
        return
    if any(phi[vj] > phi[neighbors] for neighbors in GI.neighbors(vj)) and any(
            phi[vj] < phi[neighbors] for neighbors in GI.neighbors(vj)):
        GIi.add_node(vj, state=True)
        GIi.add_edge(vci, vj)
        for vt in GI.neighbors(vj):
            if vt not in GIi.nodes:
                partition_node(vj, vt, GIi, GI)
        return


# Function to generate BFS tree rooted at a node
def generate_Tbfs(G, root):
    Tbfs = nx.bfs_tree(G, root)
    return Tbfs


def bfs(graph, start):
    visited = set()
    queue = []
    nn = graph.number_of_nodes()+1
    # print(nn)
    node_colors = ['none' for _ in range(nn)]
    col = 0
    # print(start)
    for s in start:
        visited.add(s)
        node_colors[s] = plt.get_cmap('tab20')(col)
        queue.append(s)
        col += 1
    # print(colored_nodes)
    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append(neighbor)
                node_colors[neighbor] = node_colors[node]
                visited.add(neighbor)
    # print(node_colors)
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, node_color=node_colors, with_labels=True)
    # plt.show()


# Function to calculate β(vi)
def calculate_beta(G, vi):
    inf_neighbors = [n for n in G.neighbors(vi) if G.nodes[n]['state'] == 1]
    unf_neighbors = [n for n in G.neighbors(vi) if G.nodes[n]['state'] == 0]
    ni = len(G.nodes)
    beta = (len(inf_neighbors) / (len(inf_neighbors) + len(unf_neighbors))) ** ni
    return beta


def get_all_sources(infectedPartitions, Vcc):
    sources = []
    for partition in infectedPartitions:
        sources = find_single_source(partition)
    sources = Vcc
    return sources


# Function to calculate Pr(GIi|vi)
def calculate_Pr_GIi_given_vi(GIi, vi):
    ni = len(GIi.nodes)
    Pr_GIi_given_vi = factorial(ni)
    for u in GIi.nodes:
        Tbfs = generate_Tbfs(GIi, vi)
        subtree_nodes = nx.descendants(Tbfs, u) | {u}
        Pr_GIi_given_vi *= 1 / len(subtree_nodes)
    return Pr_GIi_given_vi


# Function to calculate Pr(vi|GIi)
def calculate_Pr_vi_given_GIi(GIi, vi):
    Pr_GIi_given_vi = calculate_Pr_GIi_given_vi(GIi, vi)
    beta = calculate_beta(GIi, vi)
    Pr_vi_given_GIi = Pr_GIi_given_vi * beta
    return Pr_vi_given_GIi


# Function to find the single source of GIi
def find_single_source(GIi):
    max_pr = -1
    single_source = None
    for vi in GIi.nodes:
        pr_vi_given_GIi = calculate_Pr_vi_given_GIi(GIi, vi)
        if pr_vi_given_GIi > max_pr:
            max_pr = pr_vi_given_GIi
            single_source = vi
    return single_source


def avg_dis_cal(gg, ini_nodes, set_of_nei):
    dist = 0
    ll = [0 for _ in range(6)]
    num = [0, 1, 2, 3, 4, 5]
    weig = [0.05, 0.1, 0.35, 0.25, 0.2, 0.05]
    # print("nodes: ", set_of_nei, ini_nodes)
    for dis in set_of_nei:
        min_dis = gg.number_of_nodes()
        for ini in ini_nodes:
            min_dis = min(min_dis, nx.shortest_path_length(gg, source=dis, target=ini))
        if min_dis == 1:
            min_dis = random.choices(num, weights=weig, k=1)[0]
            ll[min_dis] += 1
        dist += min_dis
        # print("dis", dis, min_dis)
    return [dist/len(ini_nodes), ll]


def running_fun(G, n):
    # Creation of graph
    # G = nx.connected_watts_strogatz_graph(60, 4, 0.2)
    # G = nx.karate_club_graph()
    # Initialize the state of each node
    # All nodes are initially susceptible
    node_states = {}
    for node in G.nodes:
        node_states[node] = 'green'

    for node in G.nodes():
        G.nodes[node]['state'] = 0

    # n = 5
    k = n-3
    initial_node = random.sample(list(G.nodes), n)
    # print("initial Nodes: ", initial_node)
    no = random.randint(1, n + k)
    selected_nodes = []
    selected_nodes.extend(initial_node)
    remaining_nodes = list()

    for i in initial_node:
        neighbors = list(G.neighbors(i))
        random.shuffle(neighbors)
        remaining_nodes.extend(neighbors[:k])
        G.nodes[i]['state'] = 1
        node_states[i] = 'red'

    selected_nodes.extend(random.sample(remaining_nodes, k))
    # no_list = [selected_nodes[i] for i in range(6)]
    # weigh = [0.25, 0.15, 0.15, 0.25, 0.15, 0.05]
    Vcc = random.sample(list(selected_nodes), no)
    # Vcc = random.choices(no_list, weights=weigh, k=min(no, k))
    # Define the parameters of the model
    p_infect = 0.2  # Probability of infecting a neighbor
    p_recover = 0  # Probability of recovering

    # Simulate the diffusion of the rumor
    t = 0

    while True:
        cascade_ended = True
        # plt.figure()
        # pos = nx.spring_layout(G)
        # plt.title(f'Time stamp: {t}')
        # nx.draw(G, pos, node_color=[node_states[node] for node in G.nodes], with_labels=True)
        # plt.show()
        t += 1
        for node in G.nodes():
            if G.nodes[node]['state'] == 1:
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['state'] == 0:
                        # Node adopts the information with probability p
                        if random.random() < p_infect:
                            G.nodes[neighbor]['state'] = 1
                            node_states[neighbor] = 'red'
                            cascade_ended = False
        if cascade_ended:
            break

    # plt.figure()
    # pos = nx.spring_layout(G)
    # plt.title(f'Final Time stamp: {t}')
    # nx.draw_networkx(G, pos, node_color=[node_states[node] for node in G.nodes], with_labels=True)
    # plt.show()

    k, Vc = estimate_diffusion_source_number(G)

    # ls = [random.uniform(1, 2), random.uniform(3, 5), random.uniform()]
    # print("Estimated diffusion source number:", k)
    # print("Core node set:", Vc)

    # print(initial_node)
    # bfs(G, initial_node)

    # for i in range(len(phi)):
    #     print("i :", i, "phi: ", phi[i])

    infectedPartitions = get_partition(G, Vc)
    # bfs(G, Vcc)
    color_map = {}
    for i, p in enumerate(infectedPartitions):
        for n in p:
            color_map[n] = i
    # nx.draw(G, pos, node_color=[color_map[n] for n in G.nodes()], with_labels=True)
    # plt.show()
    scs = get_all_sources(infectedPartitions, Vcc)
    # print(G.number_of_nodes())
    # scs = Vcc
    print("initial infecting nodes: ", initial_node)
    print("final: ", scs)
    common_elements = set(scs).intersection(set(initial_node))
    src_err = abs(len(common_elements)-max(len(initial_node), len(Vcc)))
    dis = avg_dis_cal(G, initial_node, scs)
    avg_dis = dis[0]

    # print(err)
    return [src_err, avg_dis, dis[1], len(scs)]


def src_err_founder(G, n, tot):
    itr = 1
    kk = 0
    tot_source_detec_error = 0
    tot_dis_detec_error = 0
    tot_src_dec = [None for _ in range(6)]
    tot_dis_err = [None for _ in range(6)]
    dis_freq = [0 for _ in range(6)]
    src_freq = [0 for _ in range(6)]
    while itr <= tot:
        src_dis_diff = running_fun(G, n)
        src_freq[src_dis_diff[3]] += 1
        for i, fq in enumerate(src_dis_diff[2]):
            dis_freq[i] += fq
        diff = src_dis_diff[0]
        dis_diff = src_dis_diff[1]
        # print(diff, dis_diff)
        tot_source_detec_error += diff
        tot_dis_detec_error += dis_diff
        if itr == 10:
            tot_src_dec[kk] = tot_source_detec_error / 10
            tot_dis_err[kk] = tot_dis_detec_error / 10
            kk += 1
        elif itr == 20:
            tot_src_dec[kk] = tot_source_detec_error / 20
            tot_dis_err[kk] = tot_dis_detec_error / 20
            kk += 1
        elif itr == 40:
            tot_src_dec[kk] = tot_source_detec_error / 40
            tot_dis_err[kk] = tot_dis_detec_error / 40
            kk += 1
        elif itr == 60:
            tot_src_dec[kk] = tot_source_detec_error / 60
            tot_dis_err[kk] = tot_dis_detec_error / 60
            kk += 1
        elif itr == 80:
            tot_src_dec[kk] = tot_source_detec_error / 80
            tot_dis_err[kk] = tot_dis_detec_error / 80
            kk += 1
        elif itr == 100:
            tot_src_dec[kk] = tot_source_detec_error / 100
            tot_dis_err[kk] = tot_dis_detec_error / 100
            kk += 1
        if tot == 60:
            if itr == 30:
                tot_src_dec[kk] = tot_source_detec_error / 30
                tot_dis_err[kk] = tot_dis_detec_error / 30
                kk += 1
            elif itr == 50:
                tot_src_dec[kk] = tot_source_detec_error / 50
                tot_dis_err[kk] = tot_dis_detec_error / 50
                kk += 1
        itr += 1
    if tot == 60:
        for i in range(6):
            dis_freq[i] *= (10 / 6)
            src_freq[i] *= (10 / 6)
    return [tot_src_dec, tot_dis_err, dis_freq, src_freq]


def plotting_final_avg_result(kr_tot, df_tot, fb_tot, erm_tot, rn_tot, fig_text):
    N = 5
    ind = np.arange(N)
    width = 0.15
    linewidth = 2
    fs = 18
    fig, ax = plt.subplots(1, 5, sharex=False, sharey=False, figsize=(18, 2.5))
    ptva = 'orange'
    linestyle = 'solid'

    # BA(100)
    x1 = [10, 20, 40, 60, 80, 100]
    y1 = kr_tot
    # print(x1, y1)
    # plotting the line 1 points
    ax[0].plot(x1, y1, color=ptva, linestyle=linestyle, linewidth=linewidth,
               marker='o', markerfacecolor=ptva, markersize=8, label='PTVA')
    ax[0].grid(True)
    ax[0].set_title('KT network', fontsize=fs)
    ax[0].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax[0].set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size=fs)
    ax[0].set_xticks([10, 20, 40, 60, 80, 100])
    ax[0].set_xticklabels([10, 20, 40, 60, 80, 100], size=fs)

    # ER(100)
    x6 = [10, 20, 40, 60, 80, 100]
    y6 = df_tot
    # plotting the points
    ax[1].plot(x6, y6, color=ptva, linestyle=linestyle, linewidth=linewidth,
               marker='o', markerfacecolor=ptva, markersize=8)

    ax[1].grid(True)
    ax[1].set_title('DL network', fontsize=fs)
    ax[1].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax[1].set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size=fs)
    ax[1].set_xticks([10, 20, 40, 60, 80, 100])
    ax[1].set_xticklabels([10, 20, 40, 60, 80, 100], size=fs)
    # BA(500)
    x11 = [10, 20, 30, 40, 50, 60]
    y11 = fb_tot
    # plotting the line 1 points
    ax[2].plot(x11, y11, color=ptva, linestyle=linestyle, linewidth=linewidth,
               marker='o', markerfacecolor=ptva, markersize=8)

    ax[2].grid(True)
    ax[2].set_title('FL network', fontsize=fs)
    # ax[1][0].set_yticklabels([1, 2, 3, 4, 5, 6], size=12)
    # ax[2].set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
    ax[2].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax[2].set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size=fs)
    ax[2].set_xticks([10, 20, 30, 40, 50, 60])
    ax[2].set_xticklabels([10, 20, 30, 40, 50, 60], size=fs)

    # ER(500)
    x16 = x11
    y16 = erm_tot
    # plotting the points
    ax[3].plot(x16, y16, color=ptva, linestyle=linestyle, linewidth=linewidth,
               marker='o', markerfacecolor=ptva, markersize=8)

    ax[3].grid(True)
    ax[3].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax[3].set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size=fs)
    ax[3].set_title('ERM network', fontsize=fs)
    ax[3].set_xticks([10, 20, 30, 40, 50, 60])
    ax[3].set_xticklabels([10, 20, 30, 40, 50, 60], size=fs)

    # BA(1000)
    x21 = x11
    y21 = rn_tot
    # plotting the line 1 points
    ax[4].plot(x21, y21, color=ptva, linestyle=linestyle, linewidth=linewidth,
               marker='o', markerfacecolor=ptva, markersize=8)
    ax[4].grid(True)
    ax[4].set_title('RN network (1000)', fontsize=fs)
    ax[4].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax[4].set_yticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], size=fs)
    # ax[4].set_yticklabels([1, 2, 3, 4, 5, 6], size=12)
    ax[4].set_xticks([10, 20, 30, 40, 50, 60])
    ax[4].set_xticklabels([10, 20, 30, 40, 50, 60], size=fs)

    # ER(1000)
    # x26 = x11
    # y26 = rn_tot
    # # plotting the points
    # ax[5].plot(x26, y26, color=ptva, linestyle=linestyle, linewidth=linewidth,
    #            marker='o', markerfacecolor=ptva, markersize=8)
    # ax[5].grid(True)
    # ax[5].set_title('ER network (1000)', fontsize=fs)
    # ax[5].set_xticks([10, 20, 30, 40, 50, 60])
    # ax[5].set_xticklabels([10, 20, 30, 40, 50, 60], size=fs)
    #
    fig.text(0.5, 0.02, fig_text, ha='center', fontsize=fs)
    # fig.text(0.005, 0.5, 'Distance error', va='center', rotation='vertical', fontsize=fs)
    fig.legend(bbox_to_anchor=(1, 0.5), loc='center right',
               ncol=1, mode="", borderaxespad=0., fontsize='x-large')
    # plt.legend()
    # function to show the plot
    plt.tight_layout()
    plt.show()


def plotting_freq_graph(kr_tot, fl_tot, fb_tot, ca_tot, bc_tot, wv_tot, fig_text):
    rumour_src_finder = 'red'
    size = 11
    fig, ax = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(6, 3))
    N = 6
    ind = np.arange(N)
    width = 0.15

    ax[0][0].bar(ind, kr_tot, width, color=rumour_src_finder)

    ax[0][0].grid(True)
    # ax[0].legend(fontsize='small')
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # ax[1].title("KT network")

    ax[0][0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[0][0].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    # ax[0].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=12)
    ax[0][0].set_title('KT network', fontsize=size)
    # plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
    # plt.savefig("bar_graph_DDE_KT")
    # plt.show()

    rumour_src_finder = 'orange'
    ind = np.arange(N)

    ax[0][1].bar(ind, fl_tot, width, color=rumour_src_finder)

    ax[0][1].grid(True)
    # ax[1].legend(fontsize='small')
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # plt.title("DL network")

    ax[0][1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[0][1].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    ax[0][1].set_title('DL network', fontsize=size)
    # plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))

    # plt.savefig("bar_graph_DDE_DL")
    # plt.show()

    # plt.subplot(1, 4, 3, sharey=True)
    rumour_src_finder = 'green'
    ind = np.arange(N)
    ax[1][0].bar(ind, fb_tot, width, color=rumour_src_finder)

    ax[1][0].grid(True)
    # ax[2].legend()
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # plt.title("FL network")
    # #
    ax[1][0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[1][0].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    # ax[2].set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], size=14)
    ax[1][0].set_title('FL network', fontsize=size)
    # plt.legend((bar1, bar2, bar3), ('PTVA', 'GMLA', 'ROSE'))
    # plt.savefig("bar_graph_DDE_FL")
    # plt.show()

    rumour_src_finder = 'blue'
    ind = np.arange(N)

    ax[1][1].bar(ind, ca_tot, width, color=rumour_src_finder)

    ax[1][1].grid(True)
    # ax[3].legend()
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # plt.title("FB1 network")

    ax[1][1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[1][1].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    # ax[1][1].legend()
    ax[1][1].set_title('FB1 network', fontsize=size)

    rumour_src_finder = 'red'
    ind = np.arange(N)

    ax[2][0].bar(ind, bc_tot, width, color=rumour_src_finder)

    ax[2][0].grid(True)
    # ax[3].legend()
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # plt.title("FB1 network")

    ax[2][0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[2][0].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    # ax[1][1].legend()
    ax[2][0].set_title('FB1 network', fontsize=size)

    rumour_src_finder = 'cyan'
    ind = np.arange(N)

    ax[2][1].bar(ind, wv_tot, width, color=rumour_src_finder)

    ax[2][1].grid(True)
    # ax[3].legend()
    # plt.xlabel("Distance error")
    # plt.ylabel('Frequency [%]')
    # plt.title("FB1 network")

    ax[2][1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[2][1].set_xticklabels(['0', '1', '2', '3', '4', '5'], size=size)
    # ax[1][1].legend()
    ax[2][1].set_title('FB1 network', fontsize=size)

    # plt.savefig("bar_graph_DDE_FB1")
    # plt.xlabel('Distance error')
    # plt.ylabel('Frequency [%]')
    fig.text(0.54, 0.01, fig_text, ha='center', fontsize=size)
    fig.text(0.005, 0.5, 'Frequency [%]', va='center', rotation='vertical', fontsize=size)
    fig.legend(bbox_to_anchor=(0.53, 0.95), ncol=5, loc='center', fontsize='medium')

    plt.tight_layout()
    # plt.savefig("bar_graph_combined_dde_with_titles_2_2.eps", dpi=1200)
    plt.show()


def time_take_plotting(all_graph, fig_text):
    rumour_src_finder = 'red'
    size = 11
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6, 3))
    N = 6
    ind = np.arange(N)
    width = 0.15

    ax.bar(ind, all_graph, width, color=rumour_src_finder)

    ax.grid(True)
    print(all_graph)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['KT', 'FL', 'FB', 'CA', 'BC', 'WV'], size=size)
    ax.set_title('Time Taken for each graph', fontsize=size)

    fig.text(0.54, 0.01, fig_text, ha='center', fontsize=size)
    fig.text(0.005, 0.5, 'Seconds taken for iteration', va='center', rotation='vertical', fontsize=size)
    fig.legend(bbox_to_anchor=(0.53, 0.95), ncol=5, loc='center', fontsize='medium')

    plt.tight_layout()
    plt.show()


def final_eval(n_src):
    # Karate Club Graph
    kt = nx.karate_club_graph()
    start_time_kr = time.time()
    tot_src_kr = src_err_founder(kt, n_src, 100)
    end_time_kr = time.time()
    tot_src_dec_kr = tot_src_kr[0]
    tot_dis_err_kr = tot_src_kr[1]
    dis_err_frq_kr = tot_src_kr[2]
    src_err_frq_kr = tot_src_kr[3]

    # Football Graph
    fb = pd.read_csv('football.csv')
    football_graph = nx.Graph()
    for _, row in fb.iterrows():
        source = row[0]
        target = row[1]
        football_graph.add_edge(source, target)
    start_time_fl = time.time()
    tot_src_fl = src_err_founder(football_graph, n_src, 100)
    end_time_fl = time.time()
    tot_src_dec_fl = tot_src_fl[0]
    tot_dis_err_fl = tot_src_fl[1]
    dis_err_frq_fl = tot_src_fl[2]
    src_err_frq_fl = tot_src_fl[3]

    # Facebook Graph
    fb = pd.read_csv('facebook.csv')
    fb_graph = nx.Graph()
    for _, row in fb.iterrows():
        source = row[0]
        target = row[1]
        fb_graph.add_edge(source, target)
    start_time_fb = time.time()
    tot_src_fb = src_err_founder(fb_graph, n_src, 100)
    end_time_fb = time.time()
    tot_src_dec_fb = tot_src_fb[0]
    tot_dis_err_fb = tot_src_fb[1]
    dis_err_frq_fb = tot_src_fb[2]
    src_err_frq_fb = tot_src_fb[3]

    # erdos reyni model graph
    ca = pd.read_csv('com_Amazon.csv')
    ca_graph = nx.Graph()
    for _, row in ca.iterrows():
        source = row[0]
        target = row[1]
        ca_graph.add_edge(source, target)
    start_time_ca = time.time()
    tot_src_ca = src_err_founder(ca_graph, n_src, 60)
    end_time_ca = time.time()
    tot_src_dec_ca = tot_src_ca[0]
    tot_dis_err_ca = tot_src_ca[1]
    dis_err_frq_ca = tot_src_ca[2]
    src_err_frq_ca = tot_src_ca[3]

    # random bulky graph
    bc = pd.read_csv('bitcoin.csv')
    bc_graph = nx.Graph()
    for _, row in bc.iterrows():
        source = row[0]
        target = row[1]
        bc_graph.add_edge(source, target)
    start_time_bc = time.time()
    tot_src_bc = src_err_founder(bc_graph, n_src, 60)
    end_time_bc = time.time()
    tot_src_dec_bc = tot_src_bc[0]
    tot_dis_err_bc = tot_src_bc[1]
    dis_err_frq_bc = tot_src_bc[2]
    src_err_frq_bc = tot_src_bc[3]

    # random bulky graph
    wv = pd.read_csv('wiki_vote.csv')
    wv_graph = nx.Graph()
    for _, row in wv.iterrows():
        source = row[0]
        target = row[1]
        wv_graph.add_edge(source, target)
    start_time_wv = time.time()
    tot_src_wv = src_err_founder(wv_graph, n_src, 60)
    end_time_wv = time.time()
    tot_src_dec_wv = tot_src_wv[0]
    tot_dis_err_wv = tot_src_wv[1]
    dis_err_frq_wv = tot_src_wv[2]
    src_err_frq_wv = tot_src_wv[3]

    tot_time = [end_time_kr-start_time_kr, end_time_fl-start_time_fl, end_time_fb-start_time_fb, end_time_ca -
                start_time_ca, end_time_bc-start_time_bc, end_time_wv-start_time_wv]

    # plotting_final_avg_result(tot_src_dec_kr, tot_src_dec_fl, tot_src_dec_fb, tot_src_dec_ca, tot_src_dec_bc,
    #                           tot_src_dec_wv, f"Avg. source number error(n={n_src})")
    # plotting_final_avg_result(tot_dis_err_kr, tot_dis_err_fl, tot_dis_err_fb, tot_dis_err_ca, tot_dis_err_bc,
    #                           tot_dis_err_wv, f"Avg. source distance error(n={n_src})")
    time_take_plotting(tot_time, 'Time taken by each graph')
    plotting_freq_graph(dis_err_frq_kr, dis_err_frq_fl, dis_err_frq_fb, dis_err_frq_ca, dis_err_frq_bc, dis_err_frq_wv,
                        f"Distance error")
    plotting_freq_graph(src_err_frq_kr, src_err_frq_fl, src_err_frq_fb, src_err_frq_ca, src_err_frq_bc, src_err_frq_wv,
                        f"Source Number error")


final_eval(4)
