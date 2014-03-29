using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    internal class SenseLayer : Layer
    {
        public SenseLayer(int size)
        {
            if (size <= 0)
                throw new ArgumentOutOfRangeException("size", "must be > 0");

            for(int i = 0; i < size; i++)
            {
                Nodes.Add(new SenseNode());
            }
        }

        public void SetInput(ICollection<double> input)
        {
            if (input.Count != Size)
                throw new ArgumentOutOfRangeException("input",
                    string.Format("must be the same size as senseLayer ({0} nodes)", Nodes.Count));

            var zip = input.Zip(Nodes, (d, node) => new {input = d, Node = node});
            foreach (var pair in zip)
            {
                var senseNode = pair.Node as SenseNode;
                if (senseNode != null) senseNode.SetState(pair.input);
                else throw new InconsistentSenseLayer(pair.Node);
            }
        }
    }
}