import pydot
import os

# Ensure Graphviz is in PATH (optional, uncomment if needed)
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Adjust for your system

# Diagram 1: Simplified Human Visual Cortex Flow (V1, V2, V4, IT)
def create_visual_cortex_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', label="Schemat ludzkiej kory wzrokowej", fontsize="16")

    # Define nodes (boxes for each area)
    nodes = {
        "Input": pydot.Node("Input\n(Retina)", shape="box", style="filled", fillcolor="lightblue"),
        "V1": pydot.Node("V1\n(Podstawowe cechy:\nkrawedzie, tekstury)", shape="box", style="filled", fillcolor="lightgreen"),
        "V2": pydot.Node("V2\n(Integracja cech:\nksztalty, kontury)", shape="box", style="filled", fillcolor="lightgreen"),
        "V4": pydot.Node("V4\n(Zlozone cechy:\nkolory, wzorce)", shape="box", style="filled", fillcolor="lightgreen"),
        "IT": pydot.Node("IT\n(Rozpoznawanie obiektow)", shape="box", style="filled", fillcolor="lightgreen"),
        "Output": pydot.Node("Output\n(Percepcja wzrokowa)", shape="box", style="filled", fillcolor="lightyellow")
    }

    # Add nodes to the graph
    for node in nodes.values():
        graph.add_node(node)

    # Define edges (arrows showing flow)
    edges = [
        ("Input", "V1"),
        ("V1", "V2"),
        ("V2", "V4"),
        ("V4", "IT"),
        ("IT", "Output")
    ]

    for src, dst in edges:
        graph.add_edge(pydot.Edge(nodes[src], nodes[dst]))

    # Save the diagram with UTF-8 encoding
    graph.write_png("visual_cortex_diagram.png", encoding="utf-8")
    print("Diagram ludzkiej kory wzrokowej zapisany jako 'visual_cortex_diagram.png'")

# Diagram 2: CNN Architecture (V1->V2->V4->IT + Attention)
def create_cnn_architecture_diagram():
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', label="Architektura BioInspiredCNN", fontsize="16")

    # Define nodes (boxes for each layer), replace "→" with "->"
    nodes = {
        "Input": pydot.Node("Input\n(224x224x3)", shape="box", style="filled", fillcolor="lightblue"),
        "V1": pydot.Node("V1\n[Conv(3->32), ReLU, BN]\n[Conv(32->32), ReLU, BN]\n[MaxPool(2x2)]", shape="box", style="filled", fillcolor="lightgreen"),
        "V2": pydot.Node("V2\n[Conv(32->64), ReLU, BN]\n[Conv(64->64), ReLU, BN]\n[MaxPool(2x2)]", shape="box", style="filled", fillcolor="lightgreen"),
        "V4": pydot.Node("V4\n[Conv(64->128), ReLU, BN]\n[Conv(128->128), ReLU, BN]\n[MaxPool(2x2)]", shape="box", style="filled", fillcolor="lightgreen"),
        "Attention": pydot.Node("Attention\n[Conv(128->1), Sigmoid]\n(Multiplication)", shape="box", style="filled", fillcolor="lightcoral"),
        "IT": pydot.Node("IT\n[AdaptiveAvgPool(7x7)]\n[Flatten]\n[Linear(6272->512), ReLU]\n[Dropout(0.5)]\n[Linear(512->4)]", shape="box", style="filled", fillcolor="lightgreen"),
        "Output": pydot.Node("Output\n(4 klasy:\nnormal, diabetic_retinopathy,\ncataract, glaucoma)", shape="box", style="filled", fillcolor="lightyellow")
    }

    # Add nodes to the graph
    for node in nodes.values():
        graph.add_node(node)

    # Define edges (arrows showing flow)
    edges = [
        ("Input", "V1"),
        ("V1", "V2"),
        ("V2", "V4"),
        ("V4", "Attention"),
        ("V4", "IT"),  # Direct path to IT after attention multiplication
        ("Attention", "IT", "Multiplication"),  # Attention applied to V4 output
        ("IT", "Output")
    ]

    for edge in edges:
        if len(edge) == 2:
            src, dst = edge
            graph.add_edge(pydot.Edge(nodes[src], nodes[dst]))
        else:
            src, dst, label = edge
            graph.add_edge(pydot.Edge(nodes[src], nodes[dst], label=label))

    # Save the diagram with UTF-8 encoding
    graph.write_png("cnn_architecture_diagram.png", encoding="utf-8")
    print("Diagram architektury CNN zapisany jako 'cnn_architecture_diagram.png'")

# Run the functions to generate diagrams
if __name__ == "__main__":
    create_visual_cortex_diagram()
    create_cnn_architecture_diagram()