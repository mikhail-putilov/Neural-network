using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;

namespace NetworkTests
{
    /// <summary>
    ///     Summary description for ErrorTests
    /// </summary>
    [TestClass]
    public class ErrorTests
    {
        private readonly Layer outputLayer;
        private readonly SenseLayer sLayer;

        public ErrorTests()
        {
            sLayer = new SenseLayer(1);
            outputLayer = new Layer(3, e => e*4);
        }

        /// <summary>
        ///     Gets or sets the test context which provides
        ///     information about and functionality for the current test run.
        /// </summary>
        public TestContext TestContext { get; set; }

        #region Additional test attributes

        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //

        #endregion

        [TestMethod]
        public void TestMethod1()
        {
            sLayer.SetInput(new[] {1.0});
            Node.Connect(outputLayer.Nodes[0], sLayer.Nodes[0], 2.0);
            Node.Connect(outputLayer.Nodes[1], sLayer.Nodes[0], 3.0);
            Node.Connect(outputLayer.Nodes[2], sLayer.Nodes[0], 4.0);

            var net = new Network(sLayer, outputLayer);
            ICollection<double> actual = net.Run(new[] {2.0}.ToList());
            CollectionAssert.AreEquivalent(new[] {2.0*2.0*4.0, 3.0*2.0*4.0, 4.0*2.0*4.0}, actual.ToList());
        }
    }
}