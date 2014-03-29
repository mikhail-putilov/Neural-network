using System.Collections.Generic;
using System.Linq;
using System;

namespace NeuralNetwork
{
    public class Node
    {
        private readonly ActivationFunction _activationFunction;
        private readonly List<Connection> _parentConnections = new List<Connection>();
        private readonly List<Connection> _childConnections = new List<Connection>();

        protected double State;
        protected double Delta;

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
            State = _activationFunction(WeightFunction(_parentConnections));
            return State;
        }

        /// <summary>
        /// Add connection (link to inputNode node)
        /// </summary>
        /// <param name="con">TBA connection</param>
        private void AddParentConnection(Connection con)
        {
            _parentConnections.Add(con);
        }

        private void AddChildConnection(Connection con)
        {
            _childConnections.Add(con);
        }

        /// <summary>
        /// Connects two nodes
        /// </summary>
        /// <param name="parentNode">node that would be connected with reverse connection (usually it is inputNode node)</param>
        /// <param name="childNode">node that would be connected with forward connection</param>
        /// <param name="weight">weight of a connection</param>
        public static void Connect(Node parentNode, Node childNode, double weight)
        {
            var connection = new Connection(parentNode, childNode, weight);
            childNode.AddParentConnection(connection); //в inConnections надо смотреть на parentnode
            parentNode.AddChildConnection(connection); //в outConnections надо смотроеть на childnode
        }

        /// <summary>
        /// Linear combination of inConnections
        /// </summary>
        private static double WeightFunction(IEnumerable<Connection> connections)
        {
            return connections.Sum(conn => conn.Weight*conn.ParentNode.CalculateState());
        }

        public void CalculateDelta(Func<double, double> derivative)
        {
            Delta = derivative(State) * Error;
        }

        public void CalculateError()
        {
            Error = _childConnections.Select(connection => connection.ChildNode.Delta * connection.Weight).Sum();
        }

        public void ReweightRecursively(double learningCoef)
        {
            foreach (var connection in _parentConnections)
            {
                //connection.ParentNode.
                connection.Weight += learningCoef * connection.ParentNode.State * Delta;
                connection.ParentNode.ReweightRecursively(learningCoef);
            }
        }
    }
}