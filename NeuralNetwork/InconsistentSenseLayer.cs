﻿using System;

namespace NeuralNetwork
{
    internal class InconsistentSenseLayer : Exception
    {
        public InconsistentSenseLayer(Node node) : base(string.Format("Node with hash {0} is not SenseNode but exists in sense layer", node.GetHashCode()))
        {
        }
    }
}