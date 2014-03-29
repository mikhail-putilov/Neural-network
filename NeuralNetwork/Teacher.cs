using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    internal class Teacher
    {
        private static double learningCoef = 0.1;
        private static double EpsilonTraining = 0.1;
        private readonly Network _network;
        private readonly ICollection<KnownPrecedent> _trainingSet;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            _network = network;
            _trainingSet = trainingSet;
        }

        public void Train()
        {
            for (int i = 0; i < 10000; i++)
            {
                foreach (KnownPrecedent precedent in _trainingSet)
                {

                    //Console.WriteLine("actual");
                    ICollection<double> actual = _network.Run(precedent.ObjectFeatures);
                    //actual.ToList().ForEach(Console.WriteLine);
                    //Console.WriteLine("expected");
                    ICollection<double> expected = precedent.SupervisorySignal;
                    //expected.ToList().ForEach(Console.WriteLine);
                    //Console.WriteLine("error per output node");
                    ICollection<double> networkError = GetNetworkError(actual, expected);
                    //networkError.ToList().ForEach(Console.WriteLine);
                    _network.ReweightAllLayers(networkError, precedent.ObjectFeatures, learningCoef);
                }
            }
        }

        private static ICollection<double> GetNetworkError(ICollection<double> actual, ICollection<double> expected)
        {
            if (actual.Count != expected.Count)
                throw new ArgumentException("actual feature size is not equal to expected size");

            return actual.Zip(expected, (d, d1) => new { actual = d, expected = d1 })
                .Select(z => (z.expected - z.actual))
                .ToList();
        }
    }
}