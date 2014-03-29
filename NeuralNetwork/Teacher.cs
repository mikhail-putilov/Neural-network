using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Mime;
using System.Text;
using System.Xml.Schema;

namespace NeuralNetwork
{
    internal class Teacher
    {
        private const double learningCoef = 0.1;
        private static double EpsilonTraining = 0.001;
        private readonly Network network;
        private readonly ICollection<KnownPrecedent> trainingSet;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            this.network = network;
            this.trainingSet = trainingSet;
        }

        public void Train(int maxIterations)
        {
            for (int i = 0; i < maxIterations; i++)
            {
                var errors = new List<ICollection<double>>();
                foreach (KnownPrecedent precedent in trainingSet)
                {
                    ICollection<double> actual = network.Run(precedent.ObjectFeatures);
                    ICollection<double> expected = precedent.SupervisorySignal;
                    ICollection<double> networkError = GetNetworkError(actual, expected);
                    errors.Add(networkError);
                    network.BackPropagation(networkError, precedent.ObjectFeatures, learningCoef);
                }
                if (doesConverge(errors))
                    break;
            }
        }

        private bool doesConverge(List<ICollection<double>> errors)
        {
            double medium = errors.Select(doubles => doubles.Sum()/(double) doubles.Count).Sum()/errors.Count;
            return medium < EpsilonTraining;
        }

        private static ICollection<double> GetNetworkError(ICollection<double> actual, ICollection<double> expected)
        {
            if (actual.Count != expected.Count)
                throw new ArgumentException("actual feature size is not equal to expected size");

            return actual.Zip(expected, (a, e) => e - a).ToList();
        }
    }
}