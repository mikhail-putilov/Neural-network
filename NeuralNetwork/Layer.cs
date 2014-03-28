using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Layer
    {
        private readonly List<Node> _nodes = new List<Node>();

        public Layer(int numberOfNodes, ActivationFunction func)
        {
            for (int i = 0; i < numberOfNodes; i++)
            {
                _nodes.Add(new Node(func));
            }
        }

        protected Layer()
        {
        }

        public int Size
        {
            get { return Nodes.Count; }
        }

        public IList<Node> Nodes
        {
            get { return _nodes; }
        }

        public ICollection<double> CalculateStates()
        {
            return _nodes.Select(node => node.CalculateState()).ToList();
        }

        /// <summary>
        ///     this will be using information from inputLayer network
        /// </summary>
        /// <param name="inputLayer"></param>
        public void FullConnectionWith(Layer inputLayer)
        {
            var random = new Random((int) DateTime.Now.ToBinary());
            foreach (Node inputNode in inputLayer._nodes)
            {
                foreach (Node outputNode in _nodes)
                {
                    Node.Connect(outputNode, inputNode, random.NextDouble() - 0.5);
                }
            }
        }

        public void Reweight(IEnumerable<double> error, double learningCoef)
        {
            error.Zip(_nodes, (err, node) => new {err, node}).ToList().ForEach(obj => obj.node.Error = obj.err);
            //calculate error
            foreach (var node in _nodes)
            {
                node.CalculateError();
            }
            //update weights
            foreach (var node in _nodes)
            {
                var deltaWeight = learningCoef*node.GetState()*node.Error;
                node.Reweight(deltaWeight);
            }
        }
    }
}