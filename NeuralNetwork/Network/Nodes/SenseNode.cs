namespace NeuralNetwork.Network.Nodes
{
    public class SenseNode : Node
    {
        /// <summary>
        ///     Sense node, no connections
        /// </summary>
        public SenseNode() : base(net => net)
        {
        }

        public override double CalculateOutput()
        {
            return Output;
        }

        public void SetState(double state)
        {
            Output = state;
        }
    }
}