using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;

namespace NetworkTests
{
    [TestClass]
    public class CalculateStateTests
    {
        [TestMethod]
        public void CalculateStateTest1()
        {
            var snode = new SenseNode();
            snode.SetState(0);
            var node = new Node(e => e);

            Node.Connect(node, snode, 1);
            Assert.IsTrue(Math.Abs(node.CalculateState() - 0.0) < 0.001);
        }

        [TestMethod]
        public void CalculateStateTest2()
        {
            var snode = new SenseNode();
            snode.SetState(2);
            var node = new Node(e => e);

            Node.Connect(node, snode, 4);
            Assert.IsTrue(Math.Abs(node.CalculateState() - 8.0) < 0.001);
        }

        [TestMethod]
        public void CalculateStateTest3()
        {
            var snode1 = new SenseNode();
            snode1.SetState(2);

            var snode2 = new SenseNode();
            snode2.SetState(3);

            var node = new Node(e => e);

            Node.Connect(node, snode1, 4);
            Node.Connect(node, snode2, 3);
            Assert.IsTrue(Math.Abs(node.CalculateState() - (4.0*2.0 + 3.0*3.0)) < 0.001);
        }

        [TestMethod]
        public void CalculateStateTransition1()
        {
            var snode1 = new SenseNode();
            snode1.SetState(2);

            var node1 = new Node(e => e);

            var node2 = new Node(e => e);

            Node.Connect(node1, snode1, 4);
            Node.Connect(node2, node1, 3);
            Assert.IsTrue(Math.Abs(node2.CalculateState() - (2.0 * 4.0 * 3.0)) < 0.001);
        }

        [TestMethod]
        public void CalculateStateTransition2()
        {
            var snode1 = new SenseNode();
            snode1.SetState(1);

            var node1 = new Node(e => e*6.0);
            var node2 = new Node(e => e*7.0);

            var output = new Node(e => e);

            Node.Connect(node1, snode1, 2);
            Node.Connect(node2, snode1, 3);

            Node.Connect(output, node1, 4);
            Node.Connect(output, node2, 5);

            const double node1Output = (1.0*2.0)*6.0;
            const double node2Output = (1.0*3.0)*7.0;
            const double outputNodeoutput = node1Output*4.0 + node2Output*5.0;

            Assert.IsTrue(Math.Abs(output.CalculateState() - outputNodeoutput) < 0.001);
        }
    }
}
