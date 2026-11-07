import com.healthmarketscience.jackcess.*;
import java.io.File;
import java.util.*;
public class Dump {
  public static void main(String[] a) throws Exception {
    Database db = DatabaseBuilder.open(new File(a[0]));
    String outdir = a.length>1 ? a[1] : null;
    for (String t : db.getTableNames()) {
      Table tab = db.getTable(t);
      List<String> cols = new ArrayList<>();
      for (Column c : tab.getColumns()) cols.add(c.getName()+":"+c.getType());
      System.out.println("## "+t+" rows="+tab.getRowCount()+" cols="+cols);
      java.io.PrintWriter pw = null;
      if (outdir!=null) { pw = new java.io.PrintWriter(new File(outdir, t.replaceAll("[^A-Za-z0-9_-]","_")+".tsv"), "UTF-8");
        List<String> h=new ArrayList<>(); for (Column c: tab.getColumns()) h.add(c.getName()); pw.println(String.join("\t",h)); }
      int i=0;
      for (Row r : tab) {
        List<String> v=new ArrayList<>();
        for (Column c : tab.getColumns()) { Object o=r.get(c.getName()); String s = (o==null?"":(o instanceof byte[] ? "<bytes "+((byte[])o).length+">" : o.toString())); v.add(s.replace("\t"," ").replace("\n"," ").replace("\r"," ")); }
        if (i<3) System.out.println("   "+v);
        if (pw!=null) pw.println(String.join("\t",v));
        i++;
      }
      if (pw!=null) pw.close();
    }

  }
}
