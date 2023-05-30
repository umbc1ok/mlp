
using MLP_TAKE2;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

/*
MLP mlp = new MLP(4, 4, 3);

mlp.LoadIrisData("../../../training.data");

Console.WriteLine("TRAINING:");
mlp.Train(1000,0.01,0.9, true,true,0.01);
Console.WriteLine("FINISHED TRAINING:");
Console.WriteLine("");
Console.WriteLine("Testing learning samples:");
mlp.TestIrises();


Stream str = new FileStream("../../../network.file", FileMode.Create, FileAccess.Write);
IFormatter f = new BinaryFormatter();
f.Serialize(str, mlp);
str.Close();
str = new FileStream("../../../network.file", FileMode.Open, FileAccess.Read);
MLP mlp2 = (MLP)f.Deserialize(str);

Console.WriteLine("Tests:");
mlp2.LoadIrisData("../../../testing.data");
mlp2.TestIrises();

*/
IFormatter f = new BinaryFormatter();


MLP mlp = new MLP(4, 3, 4);
mlp.LoadAutoEncoderData("../../../autoencoder.data");
mlp.InitializeWeightsAndBiases(true);
mlp.Train(2000, 0.9, 0.2, false, true, 0.01, false);
mlp.TestIrises();

Stream str = new FileStream("../../../network.file", FileMode.Create, FileAccess.Write);

f.Serialize(str, mlp);
str.Close();


Stream str2 = new FileStream("../../../network.file", FileMode.Open, FileAccess.Read);



MLP mlp2 = (MLP)f.Deserialize(str);
mlp2.SetBiasesToZero();
mlp2.Train(2000, 0.2, 0.9, false, true, 0.01, true);



