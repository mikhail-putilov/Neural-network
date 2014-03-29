using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Network.Etc;
using NeuralNetwork.Network.Nodes;

namespace NeuralNetwork.Network.Layers
{
    public class Layer
    {
        private readonly List<Node> nodes = new List<Node>();

        public Layer(int size, ActivationFunction func)
        {
            for (int i = 0; i < size; i++)
            {
                nodes.Add(new Node(func));
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
            get { return nodes; }
        }

        public ICollection<double> CalculateStates()
        {
            return nodes.Select(node => node.CalculateOutput()).ToList();
        }

        /// <summary>
        ///     this will be using information from inputLayer network
        /// </summary>
        /// <param name="inputLayer"></param>
        public void FullConnectionWith(Layer inputLayer)
        {
            var random = new Random((int) DateTime.Now.ToBinary());
            foreach (Node parentNode in inputLayer.nodes)
            {
                foreach (Node childNode in nodes)
                {
                    Node.Connect(parentNode, childNode, random.NextDouble() - 0.5);
                }
            }
        }

        public void SetDeltaForEndLayer(IEnumerable<double> error)
        {
            //set predelta for each output node 
            error.Zip(nodes, (err, node) => new {err, node}).ToList()
                .ForEach(obj => obj.node.Predelta = obj.err);

            //calculate delta
            foreach (var node in nodes)
            {
                node.CalculateDelta(o => - o * (1 - o));
            }
        }

        public void CalculateDelta()
        {
            foreach (var node in nodes)
            {
                node.CalculatePredeltaForHidden();
                node.CalculateDelta(o =>  o * (1 - o));
            }
        }

        public void ReweightRecursively(double learningCoef)
        {
            foreach (var node in nodes)
            {
                node.ReweightRecursively(learningCoef);
            }
        }
    }
}