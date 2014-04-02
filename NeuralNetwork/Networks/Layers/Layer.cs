using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Etc;
using NeuralNetwork.Networks.Nodes;

namespace NeuralNetwork.Networks.Layers
{
    public class Layer
    {
        private readonly List<Node> _nodes = new List<Node>();

        public Layer(int size, ActivationFunction func)
        {
            for (int i = 0; i < size; i++)
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
            return _nodes.Select(node => node.CalculateOutput()).ToList();
        }

        /// <summary>
        ///     This layer will use information from input layer
        /// </summary>
        /// <param name="inputLayer">layer which is closer to a sense layer</param>
        public void FullConnectionWith(Layer inputLayer)
        {
            var random = new Random((int) DateTime.Now.ToBinary());
            foreach (Node parentNode in inputLayer._nodes)
            {
                foreach (Node childNode in _nodes)
                {
                    Node.Connect(parentNode, childNode, random.NextDouble() - 0.5);
                }
            }
        }

        public void SetDeltaForEndLayer(ICollection<double> error)
        {
            //set predelta for each output node 
            error.Zip(_nodes, (err, node) => new {err, node}).ToList()
                .ForEach(obj => obj.node.Predelta = obj.err);

            //calculate delta
            foreach (var node in _nodes)
            {
                node.CalculateDelta(o => - o * (1 - o));
            }
        }

        public void CalculateDelta()
        {
            foreach (var node in _nodes)
            {
                node.CalculatePredeltaForHidden();
                node.CalculateDelta(o =>  o * (1 - o));
            }
        }

        public void ReweightRecursively(double learningCoef)
        {
            foreach (var node in _nodes)
            {
                node.ReweightRecursively(learningCoef);
            }
        }
    }
}