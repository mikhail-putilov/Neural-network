using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Network.Training
{
    internal class Teacher
    {
        private const double LearningCoef = 0.1;
        private double epsilonTraining = 0.01;
        private readonly Network network;
        private readonly ICollection<KnownPrecedent> trainingSet;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            this.network = network;
            this.trainingSet = trainingSet;
        }

        public void Train(int maxIterations, double epsilonTraining = 0.01)
        {
            this.epsilonTraining = epsilonTraining;
            int i;
            double resultingError = double.PositiveInfinity;
            for (i = 0; i < maxIterations; i++)
            {
                var errors = new List<double>(trainingSet.Count);
                foreach (KnownPrecedent precedent in trainingSet)
                {
                    ICollection<double> actual = network.Run(precedent.ObjectFeatures);
                    ICollection<double> expected = precedent.SupervisorySignal;
                    ICollection<double> networkError = GetNetworkError(actual, expected);
                    //(cartesian normalization)^2 :
                    errors.Add(Math.Sqrt(networkError.Select(d => d*d).Sum()));
                    network.BackPropagation(networkError, precedent.ObjectFeatures, LearningCoef);
                }

                resultingError = ResultingError(errors);
                if (doesConverge(resultingError))
                    break;
            }
            Console.Out.WriteLine("Converged at {0} iteration ({1} total precedents, resulting error: {2:P})", i + 1,
                (i + 1)*trainingSet.Count, resultingError);
        }

        private bool doesConverge(double resultingError)
        {
            return resultingError < epsilonTraining;
        }

        private static double ResultingError(IEnumerable<double> errors)
        {
            double norm = Math.Sqrt(errors.Select(d => d*d).Sum());
            return norm;
        }

        private static ICollection<double> GetNetworkError(ICollection<double> actual, ICollection<double> expected)
        {
            if (actual.Count != expected.Count)
                throw new ArgumentException("actual feature size is not equal to expected size");

            return actual.Zip(expected, (a, e) => e - a).ToList();
        }
    }
}