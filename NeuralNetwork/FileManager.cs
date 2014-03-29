using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public static class FileManager
    {
        public static ICollection<KnownPrecedent> LoadPrecedencesFromFile(string filename)
        {
            var precedences = new List<KnownPrecedent>();
            using (TextReader tr = new StreamReader(filename))
            {
                string line;
                var outputSignalSize = GetOutputSignalSize(tr);
                while ((line = tr.ReadLine()) != null)
                {
                    var precedentSplit = PrecedentSplitAndCheck(line);
                    int featuresColumnSize = precedentSplit.Length - outputSignalSize;
                    var outputSignal = precedentSplit.Skip(featuresColumnSize)
                        .Take(outputSignalSize)
                        .Select(Convert.ToDouble)
                        .ToList();
                    var features = precedentSplit
                        .Take(featuresColumnSize)
                        .Select(Convert.ToDouble)
                        .ToList();
                    
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