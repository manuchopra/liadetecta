# liadetecta
Using EEG Signals For Lie Detection 

Summary :  We aim to use state of the art statistical signal processing algorithms on EEG signals for lie detection. Lie detection has recently become a topic of discussion once more. Courts of law have been interested in lie detection for a long time, but the unreliability of the polygraph (which measures various sorts of body activity such as heart rate, blood pressure, respiration, and palm
sweating) has prevented any serious use of it.
Our proposed method is built on brain waves i.e EEG signals. M achine Learning approaches to Brain-Computer Interfaces (BCI) are ideal because they allow us to interpret the neural structure of the brain without explicitly knowing it.
The P300 is a specific electrical brain wave that is triggered whenever a person sees an object familiar to him. The P300 event related potentials can be used to determine concealed knowledge that only a criminal would know. If an individual recognizes a detail of the crime, it produce a P300 ERP and he/she is likely guilty of, or at least familiar with the crime.
S c o p e : T r a i n i n g c l a s s i f i e r t o c l a s s i f y s i n g l e s e n t e n c e s t a t e m e n t s b y t e s t e r a s l i e / t r u t h . Input-Output behavior : G iven a statement, the classifier will output truth/lie.
Concrete examples of inputs and outputs :  We plan to collect our own dataset by working either in Dr. Takako’s EEG lab at CCRMA (one of our team members has worked with her before) or using a 14 channel eMotiv headset. We have devised a unique way to collect the dataset by showing people images of other people they will recognize within a group of people they wouldn’t recognize (we plan to do so by using images of their Facebook friends). The second the subject recognizes an image, we will see a P300 spike in their EEG
     
wave. This experiment is based on an industry standard experiment called Guilty Knowledge Test.
Once we have the training dataset, We will test our algorithm on yes/no statements and predict if the person lied or not. For example:
“Do you have a PhD?” -> Yes/ No “Have you ever seen Titanic?” -> Yes/ No
Evaluation metric for success: The hardest part of the project is to collect data. Apart from working at Dr. Takako’s lab and using emotiv, We also plan to reach out to other researchers who have worked on using EEG signals for lie detection to ask if they can share their datasets with us. We will evaluate our success by comparing our truth/lie predictions to the known correct answers.
