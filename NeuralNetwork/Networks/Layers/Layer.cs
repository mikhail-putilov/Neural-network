using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Etc;
using NeuralNetwork.Networks.Nodes;

namespace NeuralNetwork.Networks.Layers
{
    public class Layer
    {
        private readonly Derivative derivative;
        private readonly List<Node> nodes = new List<Node>();

        public Layer(int size, ActivationFunction func, Derivative derivative)
        {
            this.derivative = derivative;
            for (int i = 0; i < size; i++)
                nodes.Add(new Node(func));

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
        ///     This layer will use information from input layer
        /// </summary>
        /// <param name="inputLayer">layer which is closer to a sense layer</param>
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

        public void SetDeltaForEndLayer(ICollection<double> error)
        {
            //set predelta for each output node 
            error.Zip(nodes, (err, node) => new {err, node}).ToList()
                .ForEach(obj => obj.node.Predelta = obj.err);

            //calculate delta
            foreach (var node in nodes)
                node.CalculateDelta(derivative);
        }

        public void CalculateDelta()
        {
            foreach (var node in nodes)
            {
                node.CalculatePredeltaForHidden();
                node.CalculateDelta(o =>  -derivative(o));
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