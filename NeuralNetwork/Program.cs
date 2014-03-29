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
        //private const string FileData = @"..\..\precedences.csv";
        private const string FileData = @"..\..\simple.csv";

        private static void Main(string[] args)
        {
            var network = Network.PerceptronNetwork;
            try
            {
                var precedents = LoadPrecedencesFromFile(FileData);
                var teacher = new Teacher(network, precedents);
                teacher.Train();
                ICollection<double> output = network.Run(new [] {1.0}.ToList());
                output.ToList().ForEach(d => Console.Write("{0} ", d));
            }
            catch (FileNotFoundException e)
            {
                Console.WriteLine(e.Message);
            }
        }

        private static ICollection<KnownPrecedent> LoadPrecedencesFromFile(string filename)
        {
            var precedences = new List<KnownPrecedent>();
            using (TextReader tr = new StreamReader(filename))
            {
                string line;
                var outputSignalSize = GetOutputSignalSize(tr);
                while ((line = tr.ReadLine()) != null)
                {
                    var precedentSplit = PrecedentSplitAndCheck(line);

                    var outputSignal = precedentSplit.Take(outputSignalSize).Select(d => Convert.ToDouble(d, CultureInfo.InvariantCulture)).ToList();
                    var features = precedentSplit.Take(precedentSplit.Length - 1).Select(d => Convert.ToDouble(d, CultureInfo.InvariantCulture)).ToList();
                    
                    precedences.Add(new KnownPrecedent{ObjectFeatures = features, SupervisorySignal = outputSignal});
                }
            }
            return precedences;
        }

        private static string[] PrecedentSplitAndCheck(string line)
        {
            string[] precedentSplit = line.Split(',');
            if (precedentSplit.Length < 2)
            {
                throw new FormatException(
                    string.Format("line: \"{0}\" must be at least 2 items length (1 stands for input, another for output)", line));
            }
            return precedentSplit;
        }

        private static int GetOutputSignalSize(TextReader tr)
        {
            string line = tr.ReadLine();
            if (line == null)
                throw new FormatException("Given file has no indicator of the output signal size");
            try
            {
                int size = Convert.ToInt32(line);
                if (size > 0)
                    return size;
                throw new FormatException("Output signal size must be > 0");
            }
            catch (FormatException e)
            {
                throw new FormatException("Give file has corrupted indicator of the output signal size", e);
            }
        }
    }
}