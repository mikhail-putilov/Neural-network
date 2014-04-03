using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.Networks.Training
{
    internal class Teacher
    {
        private const double LearningCoef = 0.1;
        private readonly Network network;
        private readonly ICollection<KnownPrecedent> trainingSet;
        private double epsilonTraining = 0.01;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            this.network = network;
            this.trainingSet = trainingSet;
        }
        /// <summary>
        /// Resulting error must be less than this value, so typically leave it default (0.01)
        /// </summary>
        public double EpsilonTraining
        {
            get { return epsilonTraining; }
            set { epsilonTraining = value; }
        }

        public void Train(int maxIterations)
        {
            int i;
            double resultingError = double.PositiveInfinity;
            for (i = 0; i < maxIterations; i++)
            {
                var errors = new List<double>(trainingSet.Count);
                foreach (KnownPrecedent precedent in trainingSet)
                {
                    ICollection<double> actual = network.Run(precedent.ObjectFeatures);
                    ICollection<double> expected = precedent.SupervisorySignal;
                    var networkError = actual.Zip(expected, (a, e) => e - a);
                    //(cartesian normalization)^2 :
                    errors.Add(Math.Sqrt(networkError.Select(d => d*d).Sum()));
                    network.Reweight(actual, expected, LearningCoef);
                }

                resultingError = ResultingError(errors);
                if (DoesConverge(resultingError))
                    break;
            }
            Console.Out.WriteLine("Converged at {0} iteration ({1} total precedents, resulting error: {2:P})", i + 1,
                (i + 1)*trainingSet.Count, resultingError);
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

    }
}