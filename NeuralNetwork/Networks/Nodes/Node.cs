using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Etc;

namespace NeuralNetwork.Networks.Nodes
{
    public class Node
    {
        private readonly ActivationFunction _activationFunction;
        private readonly List<Link> _parentLinks = new List<Link>();
        private readonly List<Link> _childLinks = new List<Link>();

        protected double Output;
        
        private double _delta;

        private double _predelta;

        /// <summary>
        /// For output layer: Predelta = (expected_output - Output)
        /// For hidden layers: Predelta = sum for all childs (child.Delta * link.Weight )
        /// </summary>
        public double Predelta
        {
            get { return _predelta; }
            set { _predelta = value; }
        }

        /// <summary>
        /// delta = Output*(1 - Output)(Predelta)
        /// </summary>
        public double Delta
        {
            get { return _delta; }
        }

        public double CurrentOutput
        {
            get { return Output; }
        }

        public Node(ActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// Calculates and gets state of the node
        /// </summary>
        /// <returns>state</returns>
        public virtual double CalculateOutput()
        {
            Output = _activationFunction(WeightFunctionParents(_parentLinks));
            return Output;
        }

        /// <summary>
        /// Add link (link to inputNode node)
        /// </summary>
        /// <param name="con">TBA link</param>
        private void AddParentLink(Link con)
        {
            _parentLinks.Add(con);
        }

        private void AddChildLink(Link con)
        {
            _childLinks.Add(con);
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
            _delta = derivative(Output) * _predelta;
        }

        public void CalculatePredeltaForHidden()
        {
            _predelta = _childLinks.Select(link => link.ChildNode._delta * link.Weight).Sum();
        }

        public void ReweightRecursively(double learningCoef)
        {
            //from childs to parents
            foreach (var link in _parentLinks)
            {
                link.Weight -= learningCoef * link.ParentNode.Output * _delta;
                link.ParentNode.ReweightRecursively(learningCoef);
            }
        }
    }
}