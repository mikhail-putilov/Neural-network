using System;
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
        private readonly Layer oLayer;
        private readonly SenseLayer sLayer;
        private readonly Network network;

        public ErrorTests()
        {
            sLayer = new SenseLayer(size: 1);
            oLayer = new Layer(size: 1, func: e => e);
            Node.Connect(sLayer.Nodes[0], oLayer.Nodes[0], 2);
            network = new Network(sLayer, oLayer);
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
        //todo: not done
        [TestMethod]
        public void TestMethod1()
        {
            for (int i = 0; i < 5; ++i)
            {
                /**
             * graph:
             * s1 -- 2.0 -- o1
             */
                var objectFeatures = new[] {1.0};
                ICollection<double> actual = network.Run(objectFeatures);
                ICollection<double> expected = new[] {1.0};
                network.BackPropagation(actual.Zip(expected, (a, e) => e - a).ToList(), objectFeatures,
                    learningCoef: 0.1);
                Console.Out.WriteLine(oLayer.Nodes[0].Delta);
            }
        }
    }
}