using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Networks.Etc
{
    public class Cluster
    {
        public List<double> AllFeatures
        {
            get { return new[] {X, Y}.ToList(); }
        }
        public double X { get; set; }
        public double Y { get; set; }
    }
}