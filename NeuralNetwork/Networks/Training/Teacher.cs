using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Networks.Training
{
    internal class Teacher
    {
        private const double LearningCoef = 0.1;
        private readonly Network _network;
        private readonly ICollection<KnownPrecedent> _trainingSet;
        private double _epsilonTraining = 0.01;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            _network = network;
            _trainingSet = trainingSet;
        }
        /// <summary>
        /// Resulting error must be less than this value, so typically leave it default (0.01)
        /// </summary>
        public double EpsilonTraining
        {
            get { return _epsilonTraining; }
            set { _epsilonTraining = value; }
        }

        public void Train(int maxIterations, double epsilonTraining = 0.01)
        {
            EpsilonTraining = epsilonTraining;
            int i;
            double resultingError = double.PositiveInfinity;
            for (i = 0; i < maxIterations; i++)
            {
                var errors = new List<double>(_trainingSet.Count);
                foreach (KnownPrecedent precedent in _trainingSet)
                {
                    ICollection<double> actual = _network.Run(precedent.ObjectFeatures);
                    ICollection<double> expected = precedent.SupervisorySignal;
                    ICollection<double> networkError = GetNetworkError(actual, expected);
                    //(cartesian normalization)^2 :
                    errors.Add(Math.Sqrt(networkError.Select(d => d*d).Sum()));
                    _network.BackPropagation(networkError, LearningCoef);
                }

                resultingError = ResultingError(errors);
                if (DoesConverge(resultingError))
                    break;
            }
            Console.Out.WriteLine("Converged at {0} iteration ({1} total precedents, resulting error: {2:P})", i + 1,
                (i + 1)*_trainingSet.Count, resultingError);
        }

        private bool DoesConverge(double resultingError)
        {
            return resultingError < EpsilonTraining;
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