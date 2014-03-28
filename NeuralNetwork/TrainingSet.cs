using System.Collections;
using System.Collections.Generic;

namespace NeuralNetwork
{
    internal class TrainingSet : IEnumerable<KnownPrecedent>
    {
        public List<KnownPrecedent> Precedents { get; set; }

        public TrainingSet(List<KnownPrecedent> precedents)
        {
            Precedents = precedents ?? new List<KnownPrecedent>();
        }

        public IEnumerator<KnownPrecedent> GetEnumerator()
        {
            return Precedents.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}