using System.Collections.Generic;

namespace NeuralNetwork.Networks.Training
{
    public struct KnownPrecedent
    {
        public List<double> ObjectFeatures;
        public List<double> SupervisorySignal;
    }
}