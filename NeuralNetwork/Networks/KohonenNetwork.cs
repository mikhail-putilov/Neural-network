using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuralNetwork.Networks.Etc;
using NeuralNetwork.Networks.Layers;
using NeuralNetwork.Networks.Nodes;

namespace NeuralNetwork.Networks
{
    internal class KohonenNetwork : Network
    {
        protected KohonenNetwork(int numberOfClusters, int stepCoefficient)
        {
            Debug.Assert(numberOfClusters > 0, "numberOfClusters must be positive number");
            SenseLayer = new SenseLayer(2);
            Clusters = new List<Cluster>(numberOfClusters);

            EndLayer = new StepLayer(numberOfClusters, o => 1.0/(1 + Math.Exp(-o)), o => -o*(1 - o));
            EndLayer.FullConnectionWith(SenseLayer);
        }

        public List<Cluster> Clusters { get; set; }

        public void Reweight()
        {
            foreach (Cluster cluster in Clusters)
            {
                //find node with min weights in endLayer
                List<double> distances = new List<double>(EndLayer.Nodes.Count);
                foreach (Node node in EndLayer.Nodes)
                {
                    double distance = node.ParentLinks
                        .Zip(cluster.AllFeatures, (l, feature) => (l.Weight - feature) * (l.Weight - feature))
                        .Sum();
                    distances.Add(distance);
                }
                distances.s
            }
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