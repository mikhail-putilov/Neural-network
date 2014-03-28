using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    public static class Program
    {
        private const string FileData = @"..\..\precedences.csv";

        private static void Main(string[] args)
        {
            Network network = Network.TestNetworkBuilder();
//            ICollection<double> output = network.Run(new [] {2.0, 1.0}.ToList());
//            output.ToList().ForEach(d => Console.Write("{0} ", d));
            try
            {
                ICollection<KnownPrecedent> precedents = FileManager.LoadPrecedencesFromFile(FileData);
                var teacher = new Teacher(network, precedents);
                teacher.Train();
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}