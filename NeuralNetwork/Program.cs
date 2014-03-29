using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public static class Program
    {
        private const string FileDataXOR = @"..\..\XorPrecedences.csv";
        private const string FileDataNegation = @"..\..\NegationPrecedences.csv";

        private static void Main(string[] args)
        {
            XOR();
//            Negation();
        }

        private static void Negation()
        {
            var network = Network.RepeaterNetwork;
            
            var precedents = FileManager.LoadPrecedencesFromFile(FileDataNegation);
            var teacher = new Teacher(network, precedents);
            teacher.Train(10000);
            ICollection<double> result = network.Run(new[] { 1.0 });
            Console.Write("1.0 unrepeat:\t");
            result.ToList().ForEach(d => Console.Write("{0:F} ", d));

            Console.WriteLine();

            ICollection<double> result2 = network.Run(new[] { 0.0 });
            Console.Write("0.0 unrepeat:\t");
            result2.ToList().ForEach(d => Console.Write("{0:F} ", d));
            Console.WriteLine();
        }

        private static void XOR()
        {
            var network = Network.XORNetwork;
            try
            {
                var precedents = FileManager.LoadPrecedencesFromFile(FileDataXOR);
                var teacher = new Teacher(network, precedents);
                teacher.Train(100000);
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