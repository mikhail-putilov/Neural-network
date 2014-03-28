namespace NeuralNetwork
{
    public class Connection
    {
        public Connection(Node inputNode, Node outputNode, double weight)
        {
            InputNode = inputNode;
            OutputNode = outputNode;
            Weight = weight;
        }

        public Node InputNode { get; set; }
        public Node OutputNode { get; set; }
        public double Weight { get; set; }
    }
}