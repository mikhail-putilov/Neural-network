using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Network.Etc;

namespace NeuralNetwork.Network.Nodes
{
    public class Node
    {
        private readonly ActivationFunction activationFunction;
        private readonly List<Link> parentLinks = new List<Link>();
        private readonly List<Link> childLinks = new List<Link>();

        protected double Output;
        
        private double delta;

        private double predelta;

        /// <summary>
        /// For output layer: Predelta = (expected_output - Output)
        /// For hidden layers: Predelta = sum for all childs (child.Delta * link.Weight )
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
            Output = activationFunction(WeightFunctionParents(parentLinks));
            return Output;
        }

        /// <summary>
        /// Add link (link to inputNode node)
        /// </summary>
        /// <param name="con">TBA link</param>
        private void AddParentLink(Link con)
        {
            parentLinks.Add(con);
        }

        private void AddChildLink(Link con)
        {
            childLinks.Add(con);
        }

        /// <summary>
        /// Connects two nodes
        /// </summary>
        /// <param name="parentNode">node that would be connected with reverse link (usually it is inputNode node)</param>
        /// <param name="childNode">node that would be connected with forward link</param>
        /// <param name="weight">weight of a link</param>
        public static void Connect(Node parentNode, Node childNode, double weight)
        {
            var link = new Link(parentNode, childNode, weight);
            childNode.AddParentLink(link); //в inLinks надо смотреть на parentnode
            parentNode.AddChildLink(link); //в outLinks надо смотроеть на childnode
        }

        /// <summary>
        /// Linear combination of parent nodes
        /// </summary>
        private static double WeightFunctionParents(IEnumerable<Link> links)
        {
            return links.Sum(conn => conn.Weight*conn.ParentNode.CalculateOutput());
        }

        public void CalculateDelta(Func<double, double> derivative)
        {
            delta = derivative(Output) * predelta;
        }

        public void CalculatePredeltaForHidden()
        {
            predelta = childLinks.Select(link => link.ChildNode.delta * link.Weight).Sum();
        }

        public void ReweightRecursively(double learningCoef)
        {
            //from childs to parents
            foreach (var link in parentLinks)
            {
                link.Weight -= learningCoef * link.ParentNode.Output * delta;
                link.ParentNode.ReweightRecursively(learningCoef);
            }
        }
    }
}