using System.Collections.Generic;

namespace NeuralNetwork.Network.Training
{
    public struct KnownPrecedent
    {
        public List<double> ObjectFeatures;
        public List<double> SupervisorySignal;
    }
}