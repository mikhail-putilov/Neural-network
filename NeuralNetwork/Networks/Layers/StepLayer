using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetwork.Networks.Nodes;
using NeuralNetwork.Networks.Etc;

namespace NeuralNetwork.Networks.Layers
{
    class StepLayer : Layer
    {
        /// <summary>
        /// Add additional "virtual" sense node for each node in this layer with parameters:
        /// <ul>
        ///     <li>Identity activation function: net => net</li>
        ///     <li>State of the sense node is set to 1</li>
        ///     <li>Connection weight is applied by "rnd.NextDouble() - 0.5" formula</li>
        /// </ul>
        /// </summary>
        /// <param name="size">number of nodes in layer</param>
        /// <param name="activationFunction">the function is use to produce output</param>
        public StepLayer(int size, ActivationFunction activationFunction)
            : base(size, activationFunction)
        {
            Random rnd = new Random(DateTime.Now.Millisecond);
            foreach (var node in Nodes)
            {
                var virtualStepInputNode = new SenseNode();
                virtualStepInputNode.SetState(1.0);

                Node.Connect(virtualStepInputNode, node, rnd.NextDouble() - 0.5);
            }
        }
    }
}
