namespace NeuralNetwork
{
    internal class Connection
    {
        public Connection(Node parentNode, Node childNode, double weight)
        {
            ChildNode = childNode;
            ParentNode = parentNode;
            Weight = weight;
        }

        public Node ChildNode { get; set; }
        public Node ParentNode { get; set; }
        public double Weight { get; set; }
    }
}