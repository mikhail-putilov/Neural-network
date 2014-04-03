using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork.Networks;
using NeuralNetwork.Networks.Training;

namespace NeuralNetwork
{
    public static class Program
    {
        private const string FileDataXOR = @"..\..\Resources\XorPrecedences.csv";
        private const string FileDataNegation = @"..\..\Resources\NegationPrecedences.csv";
        private const int MaxIterations = 1000000;

        private static void Main(string[] args)
        {
            XOR();
//            Negation();
        }

        private static void Negation()
        {
            Network network = new Negation();
            ICollection<KnownPrecedent> precedents;
            try
            {
                precedents = FileManager.LoadPrecedencesFromFile(FileDataNegation);
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
                return;
            }
            var teacher = new Teacher(network, precedents) {EpsilonTraining = 0.01};
            teacher.Train(MaxIterations);

            //Fancy output
            Action<double> fancyRun = input =>
            {
                ICollection<double> result = network.Run(new[]{input});
                Console.Write("{0} negate :\t", input);
                result.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
            };

            fancyRun(1);
            fancyRun(0);
            fancyRun(0.5);
            fancyRun(0.7);
            fancyRun(0.2);
        }

        private static void XOR()
        {
            Network network = new XOR();
            ICollection<KnownPrecedent> precedents;
            try
            {
                precedents = FileManager.LoadPrecedencesFromFile(FileDataXOR);
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
                return;
            }
            var teacher = new Teacher(network, precedents);
            teacher.Train(MaxIterations);

            //Fancy output
            Action<double[]> fancyRun = input =>
            {
                ICollection<double> result = network.Run(input);
                Console.Write("{0} xor {1}:\t", input[0], input[1]);
                result.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
            };

            fancyRun(new[] {1.0, 0.0});
            fancyRun(new[] {0.0, 0.0});
            fancyRun(new[] {0.0, 1.0});
            fancyRun(new[] {1.0, 1.0});
        }
    }
}