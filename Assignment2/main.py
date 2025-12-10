"""
Social Network Analysis with Bot Detection and Adversarial Attacks
Dataset: Facebook Egonets from SNAP
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import requests
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class SocialNetworkAnalyzer:
    """Analyzer for social network structure and automated account detection"""
    
    def __init__(self):
        self.G = None
        self.features_df = None
        self.bot_labels = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_facebook_data(self, data_path='facebook'):
        """Load Facebook egonets dataset"""
        print("Loading Facebook egonets dataset...")
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created directory: {data_path}")
            print("Download dataset from https://snap.stanford.edu/data/facebook_combined.txt.gz")
            return None
        
        edges_file = os.path.join(data_path, 'facebook_combined.txt')
        if os.path.exists(edges_file):
            self.G = nx.read_edgelist(edges_file, nodetype=int)
            print(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        else:
            print("Creating sample network...")
            self.G = self._create_sample_network()
            
        return self.G
    
    def _create_sample_network(self, n_nodes=500, p_edge=0.02):
        """Generate synthetic network with realistic social structure"""
        G = nx.erdos_renyi_graph(n_nodes, p_edge)
        
        for i in range(0, n_nodes, 50):
            clique = list(range(i, min(i+10, n_nodes)))
            for u in clique:
                for v in clique:
                    if u != v:
                        G.add_edge(u, v)
        print(f"Generated synthetic network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def compute_graph_metrics(self):
        """Compute essential graph metrics for all nodes"""
        print("\nComputing graph metrics...")
        
        metrics = {}
        
        print("- Computing degree metrics...")
        degree_dict = dict(self.G.degree())
        metrics['degree'] = degree_dict
        metrics['avg_neighbor_degree'] = nx.average_neighbor_degree(self.G)
        
        print("- Computing clustering coefficients...")
        metrics['clustering'] = nx.clustering(self.G)
        
        print("- Computing centrality measures...")
        metrics['betweenness'] = nx.betweenness_centrality(self.G)
        metrics['closeness'] = nx.closeness_centrality(self.G)
        metrics['eigenvector'] = nx.eigenvector_centrality(self.G, max_iter=1000)
        
        print("- Computing PageRank...")
        metrics['pagerank'] = nx.pagerank(self.G)
        
        print("- Detecting communities...")
        communities = nx.community.greedy_modularity_communities(self.G)
        community_dict = {}
        for idx, community in enumerate(communities):
            for node in community:
                community_dict[node] = idx
        metrics['community'] = community_dict
        
        self.features_df = pd.DataFrame(metrics)
        
        self.features_df['degree_centrality'] = self.features_df['degree'] / (len(self.G.nodes()) - 1)
        self.features_df['clustering_x_degree'] = self.features_df['clustering'] * self.features_df['degree']
        
        print(f"Computed {len(self.features_df.columns)} features for {len(self.features_df)} nodes")
        return self.features_df
    
    def generate_bot_labels(self, bot_ratio=0.15):
        """Generate ground truth labels based on behavioral heuristics"""
        print(f"\nLabeling suspected automated accounts ({bot_ratio*100}% threshold)...")
        
        n_nodes = len(self.G.nodes())
        n_bots = int(n_nodes * bot_ratio)
        
        # Automated accounts exhibit characteristic network patterns:
        # - Lower clustering coefficient (less embedded in social circles)
        # - Higher degree (indiscriminate following behavior)
        # - Lower betweenness centrality (peripheral network position)
        
        # Compute suspicion score based on structural anomalies
        bot_score = (
            -self.features_df['clustering'] +
            0.5 * (self.features_df['degree'] / self.features_df['degree'].max()) +
            -0.3 * self.features_df['betweenness']
        )
        
        bot_nodes = bot_score.nlargest(n_bots).index.tolist()
        
        self.bot_labels = pd.Series(0, index=self.G.nodes())
        self.bot_labels[bot_nodes] = 1
        
        print(f"Labeled {self.bot_labels.sum()} accounts as automated")
        return self.bot_labels
    
    def train_baseline_model(self):
        """Train baseline bot detection model"""
        print("\n" + "="*60)
        print("TRAINING BASELINE BOT DETECTION MODEL")
        print("="*60)
        
        X = self.features_df.drop(['community'], axis=1)
        y = self.bot_labels
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        print("\nBaseline Model Performance:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Automated']))
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'y_test': y_test
        }
    
    def structural_evasion_attack(self, n_attacks=20):
        """
        Structural Evasion Attack: Modify bot nodes' connections to evade detection
        """
        print("\n" + "="*60)
        print("APPLYING STRUCTURAL EVASION ATTACK")
        print("="*60)
        
        bot_nodes = self.bot_labels[self.bot_labels == 1].index.tolist()
        attack_nodes = np.random.choice(bot_nodes, size=min(n_attacks, len(bot_nodes)), replace=False)
        
        print(f"Attacking {len(attack_nodes)} bot nodes...")
        
        for bot in attack_nodes:
            neighbors = list(self.G.neighbors(bot))
            if len(neighbors) >= 2:
                for _ in range(min(3, len(neighbors)//2)):
                    n1, n2 = np.random.choice(neighbors, size=2, replace=False)
                    if not self.G.has_edge(n1, n2):
                        self.G.add_edge(n1, n2)
            
            if len(neighbors) > 5:
                high_degree_neighbors = sorted(neighbors, key=lambda x: self.G.degree(x), reverse=True)
                for node in high_degree_neighbors[:2]:
                    self.G.remove_edge(bot, node)
            
            high_centrality_nodes = self.features_df['betweenness'].nlargest(50).index.tolist()
            available_nodes = [n for n in high_centrality_nodes if n not in neighbors and n != bot]
            if available_nodes:
                new_connection = np.random.choice(available_nodes)
                self.G.add_edge(bot, new_connection)
        
        print("Structural evasion attack completed")
        return attack_nodes
    
    def graph_poisoning_attack(self, n_attacks=30):
        """
        Graph Poisoning Attack: Inject nodes and edges to corrupt training data
        """
        print("\n" + "="*60)
        print("APPLYING GRAPH POISONING ATTACK")
        print("="*60)
        
        bot_nodes = self.bot_labels[self.bot_labels == 1].index.tolist()
        human_nodes = self.bot_labels[self.bot_labels == 0].index.tolist()
        
        max_node_id = max(self.G.nodes())
        
        print(f"Injecting {n_attacks} poisoned nodes/edges...")
        
        n_fake_nodes = n_attacks // 2
        fake_nodes = []
        
        for i in range(n_fake_nodes):
            fake_node = max_node_id + i + 1
            fake_nodes.append(fake_node)
            self.G.add_node(fake_node)
            
            bots_to_connect = np.random.choice(bot_nodes, size=min(5, len(bot_nodes)), replace=False)
            for bot in bots_to_connect:
                self.G.add_edge(fake_node, bot)
            
            humans_to_connect = np.random.choice(human_nodes, size=min(8, len(human_nodes)), replace=False)
            for human in humans_to_connect:
                self.G.add_edge(fake_node, human)
            
            self.bot_labels[fake_node] = 0
        
        n_fake_edges = n_attacks - n_fake_nodes
        high_rep_humans = self.features_df.loc[human_nodes, 'pagerank'].nlargest(30).index.tolist()
        
        for _ in range(n_fake_edges):
            bot = np.random.choice(bot_nodes)
            human = np.random.choice(high_rep_humans)
            if not self.G.has_edge(bot, human):
                self.G.add_edge(bot, human)
        
        print(f"Poisoning attack completed: {len(fake_nodes)} sybil nodes, {n_fake_edges} adversarial edges")
        return fake_nodes
    
    def visualize_graph(self, title="Social Network Graph", node_colors=None, save_path=None):
        """Visualize the graph"""
        print(f"\nVisualizing: {title}")
        
        plt.figure(figsize=(14, 10))
        
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=42)
        
        if node_colors is None:
            node_colors = ['red' if self.bot_labels.get(node, 0) == 1 else 'lightblue' 
                          for node in self.G.nodes()]
        
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, 
                              node_size=30, alpha=0.7)
        nx.draw_networkx_edges(self.G, pos, alpha=0.2, width=0.5)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Automated'),
            Patch(facecolor='lightblue', alpha=0.7, label='Legitimate')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def evaluate_model_after_attack(self, attack_name):
        """Re-evaluate model after an attack"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL AFTER {attack_name.upper()}")
        print("="*60)
        
        self.compute_graph_metrics()
        
        X = self.features_df.drop(['community'], axis=1)
        y = self.bot_labels.loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\nModel Performance After {attack_name}:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Automated']))
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'y_test': y_test
        }


def main():
    """Execute the complete analysis workflow"""
    print("="*80)
    print("SOCIAL NETWORK ANALYSIS WITH BOT DETECTION AND ADVERSARIAL ATTACKS")
    print("="*80)
    
    analyzer = SocialNetworkAnalyzer()
    
    analyzer.load_facebook_data()
    analyzer.compute_graph_metrics()
    analyzer.generate_bot_labels(bot_ratio=0.15)
    
    analyzer.visualize_graph(
        title="Original Social Network (Baseline)",
        save_path="graph_baseline.png"
    )
    
    baseline_results = analyzer.train_baseline_model()
    
    G_baseline = analyzer.G.copy()
    
    analyzer.G = G_baseline.copy()
    attack_nodes = analyzer.structural_evasion_attack(n_attacks=20)
    
    analyzer.visualize_graph(
        title="After Structural Evasion Attack",
        save_path="graph_structural_evasion.png"
    )
    
    structural_results = analyzer.evaluate_model_after_attack("Structural Evasion")
    
    analyzer.G = G_baseline.copy()
    fake_nodes = analyzer.graph_poisoning_attack(n_attacks=30)
    
    analyzer.visualize_graph(
        title="After Graph Poisoning Attack",
        save_path="graph_poisoning.png"
    )
    
    poisoning_results = analyzer.evaluate_model_after_attack("Graph Poisoning")
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Condition': ['Baseline', 'Structural Evasion', 'Graph Poisoning'],
        'Accuracy': [
            baseline_results['accuracy'],
            structural_results['accuracy'],
            poisoning_results['accuracy']
        ],
        'F1-Score': [
            baseline_results['f1_score'],
            structural_results['f1_score'],
            poisoning_results['f1_score']
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(comparison['Condition'], comparison['Accuracy'], color=['green', 'orange', 'red'])
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Detection Accuracy Under Adversarial Conditions', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(comparison['Condition'], comparison['F1-Score'], color=['green', 'orange', 'red'])
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score Under Adversarial Conditions', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("- graph_baseline.png")
    print("- graph_structural_evasion.png")
    print("- graph_poisoning.png")
    print("- performance_comparison.png")
    
    return analyzer, comparison


if __name__ == "__main__":
    analyzer, results = main()
