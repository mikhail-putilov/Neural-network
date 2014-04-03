using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuralNetwork.Networks.Layers;
using System;
using NeuralNetwork.Networks.Nodes;

namespace NeuralNetwork.Networks
{
    internal class KohonenNetwork : Network
    {
        protected KohonenNetwork(int numberOfClusters, int stepCoefficient)
        {
            Debug.Assert(numberOfClusters > 0, "numberOfClusters must be positive number");
            SenseLayer = new SenseLayer(2);

            EndLayer = new StepLayer(numberOfClusters, net => 1.0 / (1 + Math.Exp(-net)));
        }

        public override void Reweight(ICollection<double> actual, ICollection<double> expected, double learningCoef)
        {
            //find node with min weights in endLayer
            //findNodeWithMinWeight(EndLayer);
        }

        //private Node findNodeWithMinWeight(Layer endLayer) 
        //{
        //    var func = ;
        //    foreach (var node in endLayer.Nodes)
        //    {
                
        //    }
        //}

    }
}