using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork.Network.Training;

namespace NeuralNetwork
{
    public static class Program
    {
        private const string FileDataXOR = @"..\..\Resources\XorPrecedences.csv";
        private const string FileDataNegation = @"..\..\Resources\NegationPrecedences.csv";
        private const int MaxIterations = 1000000;
        private static void Main(string[] args)
        {
//            XOR();
            Negation();
        }

        private static void Negation()
        {
            Network.Network network = Network.Network.RepeaterNetwork;
            try
            {
                ICollection<KnownPrecedent> precedents = FileManager.LoadPrecedencesFromFile(FileDataNegation);
                var teacher = new Teacher(network, precedents);
                teacher.Train(MaxIterations);
                ICollection<double> result = network.Run(new[] {1.0});
                Console.Write("1.0 negate:\t");
                result.ToList().ForEach(d => Console.Write("{0:F} ", d));

                Console.WriteLine();

                ICollection<double> result2 = network.Run(new[] {0.0});
                Console.Write("0.0 negate:\t");
                result2.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
        }

        private static void XOR()
        {
            Network.Network network = Network.Network.XORNetwork;
            try
            {
                ICollection<KnownPrecedent> precedents = FileManager.LoadPrecedencesFromFile(FileDataXOR);
                var teacher = new Teacher(network, precedents);
                teacher.Train(MaxIterations);
                ICollection<double> result = network.Run(new[] {1.0, 0.0});
                Console.Write("1.0 xor 0.0:\t");
                result.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
                ICollection<double> result2 = network.Run(new[] {0.0, 1.0});
                Console.Write("0.0 xor 1.0:\t");
                result2.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
                ICollection<double> result3 = network.Run(new[] {1.0, 1.0});
                Console.Write("1.0 xor 1.0:\t");
                result3.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
                ICollection<double> result4 = network.Run(new[] {0.0, 0.0});
                Console.Write("0.0 xor 0.0:\t");
                result4.ToList().ForEach(d => Console.Write("{0:F} ", d));
                Console.WriteLine();
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}