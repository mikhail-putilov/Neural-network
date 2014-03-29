using System.Collections.Generic;
using System.Linq;
using System;

namespace NeuralNetwork
{
    public class Node
    {
        private readonly ActivationFunction activationFunction;
        private readonly List<Connection> parentConnections = new List<Connection>();
        private readonly List<Connection> childConnections = new List<Connection>();

        protected double Output;
        /// <summary>
        /// delta = Output*(1 - Output)(Predelta)
        /// </summary>
        private double delta;

        private double predelta;

        /// <summary>
        /// For output layer: Predelta = (expected_output - Output)
        /// For hidden layers: Predelta = sum for all childs (child.Delta * connection.Weight )
        /// </summary>
        public double Predelta
        {
            get { return predelta; }
            set { predelta = value; }
        }

        /// <summary>
        /// delta = Output*(1 - Output)(Predelta)
        /// </summary>
        public double Delta
        {
            get { return delta; }
        }

        public double CurrentOutput
        {
            get { return Output; }
        }

        public Node(ActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
        }

        /// <summary>
        /// Calculates and gets state of the node
        /// </summary>
        /// <returns>state</returns>
        public virtual double CalculateOutput()
        {
            Output = activationFunction(WeightFunctionParents(parentConnections));
            return Output;
        }

        /// <summary>
        /// Add connection (link to inputNode node)
        /// </summary>
        /// <param name="con">TBA connection</param>
        private void AddParentConnection(Connection con)
        {
            parentConnections.Add(con);
        }

        private void AddChildConnection(Connection con)
        {
            childConnections.Add(con);
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
        /// Linear combination of parent nodes
        /// </summary>
        private static double WeightFunctionParents(IEnumerable<Connection> connections)
        {
            return connections.Sum(conn => conn.Weight*conn.ParentNode.CalculateOutput());
        }

        public void CalculateDelta(Func<double, double> derivative)
        {
            delta = derivative(Output) * predelta;
        }

        public void CalculatePredeltaForHidden()
        {
            predelta = childConnections.Select(connection => connection.ChildNode.delta * connection.Weight).Sum();
        }

        public void ReweightRecursively(double learningCoef)
        {
            //from childs to parents
            foreach (var connection in parentConnections)
            {
                connection.Weight -= learningCoef * connection.ParentNode.Output * delta;
                connection.ParentNode.ReweightRecursively(learningCoef);
            }
        }
    }
}