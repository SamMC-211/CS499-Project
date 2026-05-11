import { Stack } from "expo-router";
import { BackgroundProvider } from "../contexts/BackgroundContext";

export default function RootLayout() {
  return (
    <BackgroundProvider>
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#0F172A' },
          headerTintColor: '#93C5FD',
          headerTitleStyle: { fontWeight: '600', fontSize: 16 },
          contentStyle: { backgroundColor: '#0F172A' },
          headerShadowVisible: false,
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="sensors/drowsiness" options={{ title: 'Detection' }} />
        <Stack.Screen name="sensors/camera" options={{ title: 'Camera' }} />
        <Stack.Screen name="sensors/accelerometer" options={{ title: 'Accelerometer' }} />
        <Stack.Screen name="sensors/debug-gallery" options={{ title: 'Debug Gallery' }} />
      </Stack>
    </BackgroundProvider>
  );
}
