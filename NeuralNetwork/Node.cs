using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Node
    {
        private readonly ActivationFunction _activationFunction;
        private readonly List<Connection> _inConnections = new List<Connection>();
        private readonly List<Connection> _errConnections = new List<Connection>();

        protected double State;
        public double Error { get; set; }

        public double GetState()
        {
            return State;
        }
        public Node(ActivationFunction activationFunction)
        {
            this._activationFunction = activationFunction;
        }

        /// <summary>
        /// Calculates and gets state of the node
        /// </summary>
        /// <returns>state</returns>
        public virtual double CalculateState()
        {
            State = _activationFunction(WeightFunction(_inConnections));
            return State;
        }

        /// <summary>
        /// Add connection (link to inputNode node)
        /// </summary>
        /// <param name="con">TBA connection</param>
        private void AddInConnection(Connection con)
        {
            _inConnections.Add(con);
        }

        private void AddOutConnection(Connection con)
        {
            _errConnections.Add(con);
        }

        /// <summary>
        /// Connects two nodes
        /// </summary>
        /// <param name="outputNode">node that would be connected with forward connection</param>
        /// <param name="inputNode">node that would be connected with reverse connection (usually it is inputNode node)</param>
        /// <param name="weight">weight of a connection</param>
        public static void Connect(Node outputNode, Node inputNode, double weight)
        {
            var connection = new Connection(inputNode, outputNode, weight);
            outputNode.AddInConnection(connection);
            inputNode.AddOutConnection(connection);
        }

        /// <summary>
        /// Linear combination of inConnections
        /// </summary>
        private static double WeightFunction(IEnumerable<Connection> connections)
        {
            return connections.Sum(conn => conn.Weight*conn.InputNode.CalculateState());
        }

        public void CalculateError()
        {
            foreach (var errConnection in _errConnections)
            {
                var inputNode = errConnection.InputNode;
                inputNode.Error = Error*errConnection.Weight*inputNode.State*
                                                (1 - inputNode.State);
            }
        }

        public void Reweight(double deltaWeight)
        {
            foreach (var connection in _inConnections)
            {
                connection.Weight += deltaWeight;
            }
        }
    }
}