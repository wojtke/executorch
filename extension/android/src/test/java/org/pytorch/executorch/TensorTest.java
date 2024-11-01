package org.pytorch.executorch;

import java.nio.ByteBuffer;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;
public class TensorTest  {
  @Test
  public void testHello() {
    EValue evalue = EValue.from(1);
    ByteBuffer bb = evalue.toByteBuffer();
    assertEquals(4, bb.get());
    assertEquals(0, bb.get());
    assertEquals(0, bb.get());
    assertEquals(0, bb.get());
    assertEquals(1, bb.get());

    // Tensor tensor = Tensor.fromBlob(new float[] {1.0f}, new long[] {1});
    // tensor.toByteBuffer();
  }
}
