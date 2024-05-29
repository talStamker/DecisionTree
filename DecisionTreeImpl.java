
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;// decision tree
	private List<String> labels;
	private List<String> attributes;
	private Map<String, List<String>> attributeValues;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	/// need addition
	DecisionTreeImpl(DataSet train) {
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		this.root = DTL(new ArrayList<Instance>(train.instances), new ArrayList<String>(this.attributes), null);
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * ONLY for extra credits
	 * 
	 * @param train: the training set
	 * @param tune:  the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		this.root = DTL(new ArrayList<Instance>(train.instances), new ArrayList<String>(this.attributes), null);
	}

	/**
	 * This function get array of probabilities and calculate their entropy
	 * 
	 * @param p-array of probabilities
	 * @return their entropy
	 */
	private double H(double[] p) {
		double ans = 0;
		for (int i = 0; i < p.length; i++) {
			if (p[i] == 0) {
				return 0;
			}
			ans += p[i] * Math.log(p[i]) / (Math.log(2));
		}
		return -ans;
	}

	/**
	 * This function calculate the probabilities for claculate H(class)
	 * 
	 * @param p0-all     the element are 0
	 * @param instances- list of examples
	 * @return P-s' for calculate H(class)
	 */
	private void p(double[] p0, List<Instance> instances) {
		for (int k = 0; k < instances.size(); k++) {
			for (int j = 0; j < this.labels.size(); j++) {
				if (this.labels.get(j).equals(instances.get(k).label)) {
					p0[j]++;
				}
			}
		}
		for (int j = 0; j < p0.length; j++) {
			p0[j] /= instances.size();
		}
	}

	/**
	 * This function claculate H(class|attribute(i)) and return the information
	 * Gain.
	 * 
	 * @param labels-         results of examples
	 * @param attributes
	 * @param instances
	 * @param i-              the index in attributes of the attribute
	 * @param hClass-H(class)
	 * @return information gain of attribute(i)
	 */
	private double informationGain(List<String> labels, List<String> attributes, List<Instance> instances, int i,
			double hClass) {
		List<String> values = this.attributeValues.get(attributes.get(i));
		double[] p1 = new double[values.size()];
		double[][] p2 = new double[values.size()][labels.size()];
		for (int j2 = 0; j2 < values.size(); j2++) {
			for (int j = 0; j < instances.size(); j++) {
				if (instances.get(j).attributes.get(i).equals(values.get(j2))) {
					p1[j2]++;
					for (int k = 0; k < labels.size(); k++) {
						if (labels.get(k).equals(instances.get(j).label)) {
							p2[j2][k]++;
						}
					}
				}
			}
		}
		double hOnConditionAttribute = 0;
		for (int j = 0; j < p1.length; j++) {
			for (int j2 = 0; j2 < p2[j].length; j2++) {
				if (p1[j] != 0) {
					p2[j][j2] /= p1[j];
				}
			}
			p1[j] /= instances.size();
			hOnConditionAttribute += p1[j] * H(p2[j]);
		}
		return hClass - hOnConditionAttribute;
	}

	/**
	 * This function calculate the information gain of all attribute, and give the
	 * attribute which has the maximal information gain.
	 * 
	 * @param attributes
	 * @param instances
	 * @return the attribute which has the maximal information gain.
	 */
	private String chooseAttribute(List<String> attributes, List<Instance> instances) {
		double max = Integer.MIN_VALUE, infoGain;
		String att = "";
		double[] p0 = new double[this.labels.size()];
		p(p0, instances);
		double hClass = H(p0);
		for (int i = 0; i < attributes.size(); i++) {
			if (max < (infoGain = informationGain(this.labels, attributes, instances, i, hClass))) {
				max = infoGain;
				att = attributes.get(i);
			}
		}
		return att;
	}

	/**
	 * This function checks if all instances have the same result (label).
	 * 
	 * @param instances
	 * @return
	 */
	private boolean isTheSameClassification(List<Instance> instances) {
		String label = instances.get(0).label;
		for (int i = 0; i < instances.size(); i++) {
			if (!label.equals(instances.get(i).label))
				return false;
		}
		return true;
	}

	/**
	 * This function pass all instances and add to examples just who have best=vi
	 * but with out the attribute best
	 * 
	 * @param examples-will update in function
	 * @param instances
	 * @param attributes
	 * @param best
	 * @param vi
	 */
	private void updateExamples(List<Instance> examples, List<Instance> instances, List<String> attributes, String best,
			String vi) {
		for (int i = 0; i < instances.size(); i++) {
			Instance e = instances.get(i);
			List<String> attributesOfE = new ArrayList<String>(e.attributes);
			if (attributesOfE.get(attributes.indexOf(best)).equals(vi)) {
				Instance e1 = new Instance();
				e1.label = e.label;
				attributesOfE.remove(vi);
				e1.attributes = attributesOfE;
				examples.add(e1);
			}
		}
	}

	/**
	 * This function build decision tree by using the algorithem DTL
	 * 
	 * @param instances
	 * @param attributes
	 * @param fatheValue
	 * @return decision tree
	 */
	private DecTreeNode DTL(List<Instance> instances, List<String> attributes, String fatheValue) {
		if (instances.isEmpty()) {
			return null;
		} else if (isTheSameClassification(instances)) {
			return new DecTreeNode(instances.get(0).label, null, fatheValue, true);
		} else if (attributes.isEmpty()) {
			return null;
		} else {
			String best = chooseAttribute(attributes, instances);
			DecTreeNode tree = new DecTreeNode(null, best, fatheValue, false);
			List<String> valueOptions = this.attributeValues.get(best);
			for (int i = 0; i < valueOptions.size(); i++) {
				List<Instance> examples = new ArrayList<Instance>();
				updateExamples(examples, instances, attributes, best, valueOptions.get(i));
				List<String> att = new ArrayList<String>();
				for (int j = 0; j < attributes.size(); j++) {
					if (!best.equals(attributes.get(j))) {
						att.add(attributes.get(j));
					}
				}
				DecTreeNode subtree = DTL(examples, att, valueOptions.get(i));
				if (subtree != null) {
					tree.addChild(subtree);
				}
			}
			return tree;
		}
	}

	/**
	 * This function prints the information gain (one in each line) for all the
	 * attributes at the root based on the training set, train.
	 */
	@Override
	public void rootInfoGain(DataSet train) {
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		double[] p0 = new double[this.labels.size()];
		p(p0, train.instances);
		double hClass = H(p0);
		for (int i = 0; i < this.attributes.size(); i++) {
			System.out.println("The information gain of the attribute " + this.attributes.get(i) + " is "
					+ +informationGain(this.labels, this.attributes, train.instances, i, hClass));
		}
	}

	/**
	 * This function predict the result(label) of the instance. This is recursive
	 * function that look for the answer in the decision tree.
	 * 
	 * @param root
	 * @param instance
	 * @return theprediction
	 */
	private String getClassify(DecTreeNode root, Instance instance) {
		if (root.label != null) {
			return root.label;
		} else {
			int i = this.getAttributeIndex(root.attribute);
			String valueOfInstance = instance.attributes.get(i);
			List<DecTreeNode> children = root.children;
			int j = 0;
			for (; j < children.size(); j++) {
				if (children.get(j).parentAttributeValue.equals(valueOfInstance)) {
					return getClassify(children.get(j), instance);
				}
			}
			return null;
		}
	}

	/**
	 * This function predict the result(label) of the instance, by using recursive
	 * function getClassify.
	 * 
	 * @param root
	 * @param instance
	 * @return theprediction
	 */
	@Override
	public String classify(Instance instance) {
		return getClassify(this.root, instance);
	}

	/**
	 * This function print the seccess rates precent od our decision tree. it will
	 * checks how many times the decision tree predict the correct result (label).
	 */
	@Override
	public void printAccuracy(DataSet test) {
		int correctCounter = 0;
		Instance e;
		for (int i = 0; i < test.instances.size(); i++) {
			e = test.instances.get(i);
			if (e.label.equals(classify(e))) {
				correctCounter++;
			}
		}
		double percent = correctCounter * 100.0 / test.instances.size();
		System.out.println("The success rates is " + percent + "%.");
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else {
			int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
			value = attributeValues.get(parent.attribute).get(attributeValueIndex);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + p.label + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + p.attribute + "?}");
			System.out.println(sb.toString());
			for (DecTreeNode child : p.children) {
				printTreeNode(child, p, k + 1);
			}
		}
	}

	/**
	 * Helper function to get the index of the label in labels list
	 */
	private int getLabelIndex(String label) {
		for (int i = 0; i < this.labels.size(); i++) {
			if (label.equals(this.labels.get(i))) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Helper function to get the index of the attribute in attributes list
	 */
	private int getAttributeIndex(String attr) {
		for (int i = 0; i < this.attributes.size(); i++) {
			if (attr.equals(this.attributes.get(i))) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Helper function to get the index of the attributeValue in the list for the
	 * attribute key in the attributeValues map
	 */
	private int getAttributeValueIndex(String attr, String value) {
		for (int i = 0; i < attributeValues.get(attr).size(); i++) {
			if (value.equals(attributeValues.get(attr).get(i))) {
				return i;
			}
		}
		return -1;
	}

}
