using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    internal class KohonenNetwork : Network
    {
        protected KohonenNetwork(int numberOfClusters, int stepCoefficient)
        {
            Debug.Assert(numberOfClusters > 0, "numberOfClusters must be positive number");
            SenseLayer = new SenseLayer(2);

            EndLayer = new Layer(numberOfClusters, net => net + stepCoefficient);
        }

        public override void Reweight(ICollection<double> actual, ICollection<double> expected, double learningCoef)
        {
            List<double> actualList = actual.ToList();
            int topIndex =
                (
                    from x
                        in actual
                    orderby x
                    select actualList.IndexOf(x)
                    ).Last();

        }
    }
}